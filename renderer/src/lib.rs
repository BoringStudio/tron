use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::{Context, Result};
use vulkanalia::vk;
use winit::window::Window;

pub use self::managers::{MeshBuffers, MeshManager};
pub use self::render_passes::{EncoderExt, MainPass, MainPassInput, Pass};
pub use self::shaders::{ShaderPreprocessor, ShaderPreprocessorScope};
pub use self::types::{
    Color, Normal, PipelineVertexInputExt, Position2, Position2UV, Position2UVColor, Position3,
    Position3NormalTangentUV, Position3NormalUV, Position3UV, Tangent, Vertex2, Vertex3, Vertex4,
    VertexAttribute, VertexLocation, VertexType, UV,
};

mod managers;
mod render_passes;
mod resource_registry;
mod shaders;
mod types;

pub struct RendererBuilder {
    window: Arc<Window>,
    app_version: (u32, u32, u32),
    validation_layer: bool,
    frames_in_flight: NonZeroUsize,
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

        Ok(Renderer {
            window: self.window,
            device,
            queue,
            surface,
            pass,
            fences,
            non_optimal_count: 0,
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
}

pub struct Renderer {
    window: Arc<Window>,
    device: gfx::Device,
    queue: gfx::Queue,
    surface: gfx::Surface,
    pass: MainPass,
    fences: Fences,
    non_optimal_count: usize,
}

impl Renderer {
    pub fn builder(window: Arc<Window>) -> RendererBuilder {
        RendererBuilder {
            window,
            app_version: (0, 0, 1),
            validation_layer: false,
            frames_in_flight: NonZeroUsize::new(2).unwrap(),
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
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

        let mut encoder = self.queue.create_encoder()?;

        {
            let _render_pass = encoder.with_render_pass(
                &mut self.pass,
                &MainPassInput {
                    max_image_count: surface_image.total_image_count(),
                    target: surface_image.image().clone(),
                },
                &self.device,
            )?;
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
}

struct Fences {
    fences: Box<[gfx::Fence]>,
    fence_index: usize,
}

impl Fences {
    fn new(device: &gfx::Device, count: NonZeroUsize) -> Result<Self> {
        let fences = (0..count.get())
            .map(|_| device.create_fence())
            .collect::<Result<Box<[_]>>>()?;

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
