use std::rc::Rc;

use anyhow::Result;
use shared::util::WithDefer;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::window::Window;

use self::base::RendererBase;
use self::command_buffer::GraphicsCommandPool;
use self::pipeline::Pipeline;
use self::swapchain::{Swapchain, SwapchainFramebuffer};
use self::sync::{Fence, Semaphore};

mod base;
mod command_buffer;
mod pipeline;
mod pipeline_layout;
mod shader_module;
mod swapchain;
mod sync;

pub struct RendererConfig {
    pub app_name: String,
    pub app_version: (u32, u32, u32),
    pub validation_layer_enabled: bool,
}

pub struct Renderer {
    base: Rc<RendererBase>,
    swapchain: Swapchain,
    pipeline: Pipeline,
    swapchain_framebuffers: Vec<SwapchainFramebuffer>,
    frames: Frames,
    graphics_command_pool: GraphicsCommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl Renderer {
    pub unsafe fn new(window: &Window, config: RendererConfig) -> Result<Self> {
        const MAX_FRAMES_IN_FLIGHT: usize = 2;

        let base = Rc::new(RendererBase::new(window, config)?);

        let mut swapchain = Swapchain::uninit(base.clone());
        swapchain.recreate(window)?;

        let pipeline = Pipeline::new(base.clone())?;

        let make_semaphore = || Semaphore::new(base.clone());
        let render_finished_semaphores = [make_semaphore()?, make_semaphore()?];
        let image_available_semaphores = [make_semaphore()?, make_semaphore()?];
        let make_fence = || Fence::new(base.clone(), true);
        let in_flight_fences = [make_fence()?, make_fence()?];

        let swapchain_framebuffers =
            swapchain.make_framebuffers(pipeline.render_pass().handle())?;

        let frames = Frames::new(base.clone(), MAX_FRAMES_IN_FLIGHT)?;

        let graphics_command_pool = GraphicsCommandPool::new(base.clone())?;

        let command_buffers =
            graphics_command_pool.allocate_command_buffers(swapchain.image_views().len())?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(swapchain.extent());

        let clear_color_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        for (i, command_buffer) in command_buffers.iter().enumerate() {
            let device = base.device();

            //
            let inheritance = vk::CommandBufferInheritanceInfo::builder();
            let info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::empty())
                .inheritance_info(&inheritance);
            device.begin_command_buffer(*command_buffer, &info)?;

            //
            let viewport = vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(render_area.extent.width as f32)
                .height(render_area.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(*command_buffer, 0, std::slice::from_ref(&viewport));
            device.cmd_set_scissor(*command_buffer, 0, std::slice::from_ref(&render_area));

            //
            let info = vk::RenderPassBeginInfo::builder()
                .render_pass(pipeline.render_pass().handle())
                .framebuffer(swapchain_framebuffers[i].handle())
                .render_area(render_area)
                .clear_values(std::slice::from_ref(&clear_color_value));
            device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);

            //
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            );

            //
            device.cmd_draw(*command_buffer, 3, 1, 0, 0);

            //
            device.cmd_end_render_pass(*command_buffer);

            //
            device.end_command_buffer(*command_buffer)?;
        }

        Ok(Self {
            base,
            swapchain,
            pipeline,
            frames,
            swapchain_framebuffers,
            graphics_command_pool,
            command_buffers,
        })
    }

    pub unsafe fn wait_idle(&self) -> Result<()> {
        self.base.device().device_wait_idle()?;
        Ok(())
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.frames.in_flight_fence();

        self.base
            .device()
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let (image_index, _) = self.base.device().acquire_next_image_khr(
            self.swapchain.handle(),
            u64::MAX,
            self.frames.image_available(),
            vk::Fence::null(),
        )?;
        let image_index = image_index as usize;

        self.swapchain
            .wait_for_image_fence(image_index, in_flight_fence)?;

        let wait_semaphores = &[self.frames.image_available()];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.frames.render_finished()];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.base.device().reset_fences(&[in_flight_fence])?;

        self.base.device().queue_submit(
            self.base.queues().graphics_queue,
            std::slice::from_ref(&submit_info),
            in_flight_fence,
        )?;

        let swapchains = &[self.swapchain.handle()];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.base
            .device()
            .queue_present_khr(self.base.queues().present_queue, &present_info)?;

        self.frames.next_frame();
        Ok(())
    }
}

struct Frames {
    current: usize,
    image_available_semaphores: Vec<Semaphore>,
    render_finished_semaphores: Vec<Semaphore>,
    in_flight_fences: Vec<Fence>,
}

impl Frames {
    unsafe fn new(base: Rc<RendererBase>, count: usize) -> Result<Self> {
        let make_semaphore = || Semaphore::new(base.clone());

        let image_available_semaphores = (0..count)
            .map(|_| make_semaphore())
            .collect::<Result<Vec<_>>>()?;

        let render_finished_semaphores = (0..count)
            .map(|_| make_semaphore())
            .collect::<Result<Vec<_>>>()?;

        let in_flight_fences = (0..count)
            .map(|_| Fence::new(base.clone(), true))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            current: 0,
            render_finished_semaphores,
            image_available_semaphores,
            in_flight_fences,
        })
    }

    pub fn next_frame(&mut self) {
        self.current = (self.current + 1) % self.image_available_semaphores.len();
    }

    fn image_available(&self) -> vk::Semaphore {
        self.image_available_semaphores[self.current].handle()
    }

    fn render_finished(&self) -> vk::Semaphore {
        self.render_finished_semaphores[self.current].handle()
    }

    fn in_flight_fence(&self) -> vk::Fence {
        self.in_flight_fences[self.current].handle()
    }
}
