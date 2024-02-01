use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use bumpalo::Bump;
use glam::{Mat4, Vec3};
use shared::util::DeallocOnDrop;

use crate::pipelines::{CachedGraphicsPipeline, OpaqueMeshPipeline, RenderPassEncoderExt};
use crate::render_passes::{EncoderExt, MainPass, MainPassInput};
use crate::types::CameraProjection;
use crate::RendererState;

pub struct RendererWorker {
    state: Arc<RendererState>,

    pass: MainPass,
    pipeline: CachedGraphicsPipeline,

    fences: Fences,
    surface: gfx::Surface,

    alloc: Bump,
    non_optimal_count: usize,
    started_at: Instant,
    prev_frame_at: Instant,
    frame: u32,
}

impl RendererWorker {
    pub fn new(state: Arc<RendererState>, surface: gfx::Surface) -> Result<Self> {
        const FRAMES_IN_FLIGHT: usize = 2;

        let fences = Fences::new(&state.device, FRAMES_IN_FLIGHT)?;

        let pass = MainPass::default();
        let pipeline = OpaqueMeshPipeline::make_descr(
            &state.device,
            &state.shader_preprocessor,
            &state.frame_resources,
            &state.bindless_resources,
        )
        .map(CachedGraphicsPipeline::new)?;

        Ok(Self {
            state,
            pass,
            pipeline,
            fences,
            surface,
            non_optimal_count: 0,
            alloc: Bump::default(),
            started_at: Instant::now(),
            prev_frame_at: Instant::now(),
            frame: 0,
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

        {
            profiling::scope!("eval_instructions");
            self.state.eval_instructions(&mut encoder)?;
        }

        if self
            .state
            .window_resized
            .compare_exchange(true, false, Ordering::Release, Ordering::Relaxed)
            .is_ok()
        {
            let window_size = self.state.window.inner_size();
            self.state
                .frame_resources
                .set_render_resolution(window_size.width, window_size.height);
        }

        let prev_frame_at = std::mem::replace(&mut self.prev_frame_at, Instant::now());
        let time = self.started_at.elapsed().as_secs_f32();
        let delta_time = self
            .prev_frame_at
            .duration_since(prev_frame_at)
            .as_secs_f32();

        // TEMP
        let camera_transform = Mat4::from_translation(-Vec3::new(0.0, 0.0, 3.0));
        self.state
            .frame_resources
            .set_camera(&camera_transform, &CameraProjection::default());

        let globals_dynamic_offset = self
            .state
            .frame_resources
            .flush(time, delta_time, self.frame);

        encoder.bind_graphics_descriptor_sets(
            &self.pipeline.descr().layout,
            0,
            &[
                self.state.frame_resources.descriptor_set(),
                self.state.bindless_resources.descriptor_set(),
            ],
            &[globals_dynamic_offset],
        );

        self.state.mesh_manager.bind_index_buffer(&mut encoder);

        encoder.memory_barrier(
            gfx::PipelineStageFlags::COMPUTE_SHADER | gfx::PipelineStageFlags::TRANSFER,
            gfx::AccessFlags::SHADER_WRITE | gfx::AccessFlags::TRANSFER_WRITE,
            gfx::PipelineStageFlags::VERTEX_SHADER,
            gfx::AccessFlags::SHADER_READ,
        );

        {
            profiling::scope!("opaque_mesh_render_pass");

            let mut render_pass = encoder.with_render_pass(
                &mut self.pass,
                &MainPassInput {
                    max_image_count: surface_image.total_image_count(),
                    target: surface_image.image().clone(),
                },
                device,
            )?;

            render_pass.bind_cached_graphics_pipeline(&mut self.pipeline, device)?;

            // TEMP
            let vertex_buffer_handle = self.state.mesh_manager.vertex_buffer_handle();

            let managers = self.state.synced_managers.lock().unwrap();

            let dt = managers
                .time_manager
                .compute_interpolation_factor(self.prev_frame_at);

            if let Some((objects, material_buffer_handle)) = managers
                .object_manager
                .iter_static_objects::<crate::DebugMaterial>()
                .and_then(|iter| {
                    let materials_buffer_handle = managers
                        .material_manager
                        .materials_data_buffer_handle::<crate::DebugMaterial>()?;
                    Some((iter, materials_buffer_handle))
                })
            {
                render_pass.push_constants(
                    &self.pipeline.descr().layout,
                    gfx::ShaderStageFlags::ALL,
                    0,
                    &[
                        vertex_buffer_handle.index(),
                        objects.buffer_handle().index(),
                        material_buffer_handle.index(),
                    ],
                );

                for (slot, object) in objects {
                    render_pass.draw_indexed(
                        object.first_index..object.first_index + object.index_count,
                        0,
                        slot..slot + 1,
                    );
                }
            }
        }

        encoder.image_barriers(
            gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            gfx::PipelineStageFlags::BOTTOM_OF_PIPE,
            &[gfx::ImageMemoryBarrier {
                image: surface_image.image(),
                src_access: gfx::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access: gfx::AccessFlags::empty(),
                old_layout: Some(gfx::ImageLayout::ColorAttachmentOptimal),
                new_layout: gfx::ImageLayout::Present,
                family_transfer: None,
                subresource_range: gfx::ImageSubresourceRange::whole(surface_image.image().info()),
            }],
        );

        let [wait, signal] = surface_image.wait_signal();

        {
            profiling::scope!("queue_submit");
            queue.submit(
                &mut [(gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, wait)],
                Some(encoder.finish()?),
                &mut [signal],
                Some(fence),
                &mut DeallocOnDrop(&mut self.alloc),
            )?;
        }

        let mut is_optimal = surface_image.is_optimal();
        {
            profiling::scope!("queue_present");

            self.state.window.pre_present_notify();
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
        self.frame += 1;
        Ok(())
    }
}

struct Fences {
    fences: Box<[gfx::Fence]>,
    fence_index: usize,
}

impl Fences {
    fn new(device: &gfx::Device, count: usize) -> Result<Self, gfx::OutOfDeviceMemory> {
        assert!(count > 0, "frames in flight must be greater than 0");

        let fences = (0..count)
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
