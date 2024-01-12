use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::{Context, Result};
use shared::Embed;
use vulkanalia::vk;
use winit::window::Window;

pub use self::render_passes::{EncoderExt, MainPass, MainPassInput, Pass};
pub use self::shader_preprocessor::{ShaderPreprocessor, ShaderPreprocessorScope};
pub use self::types::{
    BoundingSphere, Camera, CameraProjection, Color, CubeMeshGenerator, Frustum, Mesh, MeshBuilder,
    MeshGenerator, MeshHandle, Normal, Plane, PlaneMeshGenerator, Position, Tangent,
    VertexAttribute, VertexAttributeData, VertexAttributeKind, UV0,
};

use self::managers::MeshManager;
use self::pipelines::{CachedGraphicsPipeline, OpaqueMeshPipeline, RenderPassEncoderExt};
use self::resource_handle::ResourceHandleAllocator;

mod managers;
mod pipelines;
mod render_passes;
mod resource_handle;
mod shader_preprocessor;
mod types;

pub struct RendererBuilder {
    window: Arc<Window>,
    app_version: (u32, u32, u32),
    validation_layer: bool,
    frames_in_flight: NonZeroUsize,
    optimize_shaders: bool,
}

impl RendererBuilder {
    pub fn build(self) -> Result<Renderer> {
        let app_version = (0, 0, 1);

        gfx::Graphics::set_init_config(gfx::InstanceConfig {
            app_name: self.window.title().into(),
            app_version,
            validation_layer_enabled: self.validation_layer,
        });
        let pass = MainPass::default();

        let graphics = gfx::Graphics::get_or_init()?;
        let physical = graphics.get_physical_devices()?.find_best()?;
        let (device, queue) = physical.create_device(
            &[gfx::DeviceFeature::SurfacePresentation],
            gfx::SingleQueueQuery::GRAPHICS,
        )?;

        let mut surface = device.create_surface(self.window.clone())?;
        surface.configure()?;

        let fences = Fences::new(&device, self.frames_in_flight)?;

        let mut shader_preprocessor = ShaderPreprocessor::new();
        shader_preprocessor.set_optimizations_enabled(self.optimize_shaders);
        for (path, contents) in Shaders::iter() {
            let contents = std::str::from_utf8(contents)
                .with_context(|| anyhow::anyhow!("invalid shader {path}"))?;
            shader_preprocessor.add_file(path, contents)?;
        }

        let opaque_mesh_pipeline =
            OpaqueMeshPipeline::make_descr(&device, &mut shader_preprocessor)
                .map(CachedGraphicsPipeline::new)?;

        let mesh_manager = MeshManager::new(&device)?;
        let mesh_handle_allocator = ResourceHandleAllocator::default();

        Ok(Renderer {
            mesh_manager,
            mesh_handle_allocator,

            opaque_mesh_pipeline,
            pass,

            window: self.window,
            fences,
            non_optimal_count: 0,

            encoder: None,

            queue,
            surface,
            device,
        })
    }

    pub fn app_version(mut self, app_version: (u32, u32, u32)) -> Self {
        self.app_version = app_version;
        self
    }

    pub fn validation_layer(mut self, validation_layer: bool) -> Self {
        self.validation_layer = validation_layer;
        self
    }

    pub fn frames_in_flight(mut self, frames_in_flight: usize) -> Self {
        self.frames_in_flight = frames_in_flight.try_into().unwrap();
        self
    }

    pub fn optimize_shaders(mut self, optimize_shaders: bool) -> Self {
        self.optimize_shaders = optimize_shaders;
        self
    }
}

pub struct Renderer {
    mesh_manager: MeshManager,
    mesh_handle_allocator: ResourceHandleAllocator<Mesh>,

    // TODO: replace with render graph
    opaque_mesh_pipeline: CachedGraphicsPipeline,
    pass: MainPass,

    window: Arc<Window>,
    fences: Fences,
    non_optimal_count: usize,

    encoder: Option<gfx::Encoder>,

    queue: gfx::Queue,
    surface: gfx::Surface,

    // NOTE: device must be dropped last
    device: gfx::Device,
}

impl Renderer {
    pub fn builder(window: Arc<Window>) -> RendererBuilder {
        RendererBuilder {
            window,
            app_version: (0, 0, 1),
            validation_layer: false,
            frames_in_flight: NonZeroUsize::new(2).unwrap(),
            optimize_shaders: true,
        }
    }

    pub fn wait_idle(&self) -> Result<(), gfx::DeviceLost> {
        self.device.wait_idle()
    }

    pub fn draw(&mut self) -> Result<()> {
        let fence = {
            profiling::scope!("idle");
            self.fences.wait_next(&self.device)?
        };
        profiling::scope!("frame");

        let mut surface_image = {
            profiling::scope!("aquire_image");
            self.surface.aquire_image()?
        };

        let mut encoder = self.queue.create_primary_encoder()?;

        if let Some(secondary) = self.encoder.take() {
            encoder.execute_commands(std::iter::once(secondary.finish()?));
        }

        self.mesh_manager.buffers().bind_index_buffer(&mut encoder);

        {
            let mut render_pass = encoder.with_render_pass(
                &mut self.pass,
                &MainPassInput {
                    max_image_count: surface_image.total_image_count(),
                    target: surface_image.image().clone(),
                },
                &self.device,
            )?;

            render_pass
                .bind_cached_graphics_pipeline(&mut self.opaque_mesh_pipeline, &self.device)?;
            render_pass.draw(0..3, 0..1);
        }

        let [wait, signal] = surface_image.wait_signal();

        {
            profiling::scope!("queue_submit");
            self.queue.submit(
                &mut [(gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, wait)],
                Some(encoder.finish()?),
                &mut [signal],
                Some(fence),
            )?;
        }

        let mut is_optimal = surface_image.is_optimal();
        {
            profiling::scope!("queue_present");

            self.window.pre_present_notify();
            match self.queue.present(surface_image)? {
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
            self.device.wait_idle()?;

            self.surface.update()?;

            self.non_optimal_count = 0;
        }

        profiling::finish_frame!();
        Ok(())
    }

    pub fn add_mesh(&mut self, mesh: &Mesh) -> Result<MeshHandle> {
        let encoder = match &mut self.encoder {
            Some(encoder) => encoder,
            None => {
                let encoder = self.queue.create_secondary_encoder()?;
                self.encoder.get_or_insert(encoder)
            }
        };

        let mesh = self.mesh_manager.upload_mesh(&self.device, encoder, mesh)?;

        let handle = self.mesh_handle_allocator.alloc();
        self.mesh_manager.insert(&handle, mesh);

        Ok(handle)
    }
}

struct Fences {
    fences: Box<[gfx::Fence]>,
    fence_index: usize,
}

impl Fences {
    fn new(device: &gfx::Device, count: NonZeroUsize) -> Result<Self> {
        let fences = (0..count.get())
            .map(|_| device.create_fence())
            .collect::<Result<Box<[_]>, _>>()?;

        Ok(Self {
            fences,
            fence_index: 0,
        })
    }

    fn wait_next(&mut self, device: &gfx::Device) -> Result<&mut gfx::Fence> {
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

trait PhysicalDevicesExt {
    fn find_best(self) -> Result<gfx::PhysicalDevice>;
}

impl PhysicalDevicesExt for Vec<gfx::PhysicalDevice> {
    fn find_best(mut self) -> Result<gfx::PhysicalDevice> {
        let mut result = None;

        for (index, physical_device) in self.iter().enumerate() {
            let properties = physical_device.properties();

            let mut score = 0usize;
            match properties.v1_0.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => score += 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => score += 100,
                vk::PhysicalDeviceType::CPU => score += 10,
                vk::PhysicalDeviceType::VIRTUAL_GPU => score += 1,
                _ => continue,
            }

            tracing::info!(
                name = %properties.v1_0.device_name,
                ty = ?properties.v1_0.device_type,
                "found physical device",
            );

            match &result {
                Some((_index, best_score)) if *best_score >= score => continue,
                _ => result = Some((index, score)),
            }
        }

        let (index, _) = result.context("no suitable physical device found")?;
        Ok(self.swap_remove(index))
    }
}

const NON_OPTIMAL_LIMIT: usize = 100;

shared::embed!(
    Shaders("../../assets/shaders") = [
        "math/color.glsl",
        "math/const.glsl",
        "triangle.vert",
        "triangle.frag"
    ]
);
