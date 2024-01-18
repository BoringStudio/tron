use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::{Context, Result};
use bumpalo::Bump;
use shared::util::DeallocOnDrop;
use shared::Embed;

use crate::pipelines::{CachedGraphicsPipeline, OpaqueMeshPipeline, RenderPassEncoderExt};
use crate::render_passes::{EncoderExt, MainPass, MainPassInput};
use crate::shader_preprocessor::ShaderPreprocessor;
use crate::RendererState;

pub trait RendererWorkerCallbacks: Send + Sync + 'static {
    fn before_present(&self);
}

pub struct RendererWorkerConfig {
    pub frames_in_flight: NonZeroUsize,
    pub optimize_shaders: bool,
}

pub struct RendererWorker {
    state: Arc<RendererState>,
    callbacks: Box<dyn RendererWorkerCallbacks>,

    pass: MainPass,
    pipeline: CachedGraphicsPipeline,

    fences: Fences,
    surface: gfx::Surface,

    alloc: Bump,
    non_optimal_count: usize,
}

impl RendererWorker {
    pub fn new(
        state: Arc<RendererState>,
        config: RendererWorkerConfig,
        callbacks: Box<dyn RendererWorkerCallbacks>,
        surface: gfx::Surface,
    ) -> Result<Self> {
        let fences = Fences::new(&state.device, config.frames_in_flight)?;

        let mut shader_preprocessor = ShaderPreprocessor::new();
        shader_preprocessor.set_optimizations_enabled(config.optimize_shaders);
        for (path, contents) in Shaders::iter() {
            let contents = std::str::from_utf8(contents)
                .with_context(|| anyhow::anyhow!("invalid shader {path}"))?;
            shader_preprocessor.add_file(path, contents)?;
        }

        let pass = MainPass::default();
        let pipeline = OpaqueMeshPipeline::make_descr(&state.device, &mut shader_preprocessor)
            .map(CachedGraphicsPipeline::new)?;

        Ok(Self {
            state,
            callbacks,
            pass,
            pipeline,
            fences,
            surface,
            non_optimal_count: 0,
            alloc: Bump::default(),
        })
    }

    pub fn draw(&mut self) -> Result<()> {
        let device = &self.state.device;
        let queue = &self.state.queue;

        let fence = {
            profiling::scope!("idle");
            self.fences.wait_next(device)?
        };
        profiling::scope!("frame");

        let mut surface_image = {
            profiling::scope!("aquire_image");
            self.surface.aquire_image()?
        };

        let mut encoder = queue.create_primary_encoder()?;

        if let Some(secondary) = self.state.mesh_manager.drain() {
            encoder.execute_commands(std::iter::once(secondary.finish()?));
        }
        self.state.eval_instructions();

        self.state.mesh_manager.bind_index_buffer(&mut encoder);

        {
            let mut render_pass = encoder.with_render_pass(
                &mut self.pass,
                &MainPassInput {
                    max_image_count: surface_image.total_image_count(),
                    target: surface_image.image().clone(),
                },
                device,
            )?;

            render_pass.bind_cached_graphics_pipeline(&mut self.pipeline, device)?;
            render_pass.draw(0..3, 0..1);
        }

        let [wait, signal] = surface_image.wait_signal();

        {
            profiling::scope!("queue_submit");
            queue.submit(
                &mut [(gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, wait)],
                Some(encoder.finish()?),
                &mut [signal],
                Some(fence),
                &mut *DeallocOnDrop(&mut self.alloc),
            )?;
        }

        let mut is_optimal = surface_image.is_optimal();
        {
            profiling::scope!("queue_present");

            self.callbacks.before_present();
            match queue.present(surface_image)? {
                gfx::PresentStatus::Ok => {}
                gfx::PresentStatus::Suboptimal => is_optimal = false,
                gfx::PresentStatus::OutOfDate => {
                    is_optimal = false;
                    self.non_optimal_count += NON_OPTIMAL_LIMIT;
                }
            }
        }

        self.non_optimal_count += !is_optimal as usize;
        if self.non_optimal_count >= NON_OPTIMAL_LIMIT {
            profiling::scope!("recreate_swapchain");

            // Wait for the device to be idle before recreating the swapchain.
            device.wait_idle()?;

            self.surface.update()?;
            self.non_optimal_count = 0;
        }

        profiling::finish_frame!();
        Ok(())
    }
}

struct Fences {
    fences: Box<[gfx::Fence]>,
    fence_index: usize,
}

impl Fences {
    fn new(device: &gfx::Device, count: NonZeroUsize) -> Result<Self, gfx::OutOfDeviceMemory> {
        let fences = (0..count.get())
            .map(|_| device.create_fence())
            .collect::<Result<Box<[_]>, _>>()?;

        Ok(Self {
            fences,
            fence_index: 0,
        })
    }

    fn wait_next(&mut self, device: &gfx::Device) -> Result<&mut gfx::Fence, gfx::DeviceLost> {
        let fence_count = self.fences.len();
        let fence = &mut self.fences[self.fence_index];
        self.fence_index = (self.fence_index + 1) % fence_count;

        if !fence.state().is_unsignalled() {
            device.wait_fences(&mut [fence], true)?;
            device.reset_fences(&mut [fence])?;
        }

        Ok(fence)
    }
}

const NON_OPTIMAL_LIMIT: usize = 100;

shared::embed!(
    Shaders("../../../assets/shaders") = [
        "math/color.glsl",
        "math/const.glsl",
        "triangle.vert",
        "triangle.frag"
    ]
);
