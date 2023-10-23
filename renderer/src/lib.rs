use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::window::Window;

use self::base::RendererBase;
use self::command_buffer::GraphicsCommandPool;
use self::pipeline::{Pipeline, SurfaceDescription};
use self::swapchain::{Swapchain, SwapchainFramebuffer};
use self::sync::{Fence, Semaphore};

mod alloc;
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
    graphics_command_pool: GraphicsCommandPool,
    state: Option<RendererState>,
    resized: bool,
}

struct RendererState {
    swapchain: Swapchain,
    swapchain_framebuffers: Vec<SwapchainFramebuffer>,
    pipeline: Pipeline,
    frames: Frames,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl Renderer {
    pub unsafe fn new(window: &Window, config: RendererConfig) -> Result<Self> {
        let base = Rc::new(RendererBase::new(window, config)?);
        let graphics_command_pool = GraphicsCommandPool::new(base.clone())?;

        let state = Self::make_renderer_state(&base, &graphics_command_pool, window)?;

        Ok(Self {
            base,
            graphics_command_pool,
            state: Some(state),
            resized: false,
        })
    }

    unsafe fn make_renderer_state(
        base: &Rc<RendererBase>,
        graphics_command_pool: &GraphicsCommandPool,
        window: &Window,
    ) -> Result<RendererState> {
        const MAX_FRAMES_IN_FLIGHT: usize = 2;

        let mut swapchain = Swapchain::uninit(base.clone());
        swapchain.recreate(window)?;

        let pipeline = Pipeline::new(
            base.clone(),
            &SurfaceDescription {
                extent: swapchain.extent(),
                format: swapchain.format(),
            },
        )?;

        let swapchain_framebuffers =
            swapchain.make_framebuffers(pipeline.render_pass().handle())?;

        let frames = Frames::new(base.clone(), MAX_FRAMES_IN_FLIGHT)?;

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

        Ok(RendererState {
            swapchain,
            swapchain_framebuffers,
            pipeline,
            frames,
            command_buffers,
        })
    }

    pub fn mark_resized(&mut self) {
        self.resized = true;
    }

    pub unsafe fn wait_idle(&self) -> Result<()> {
        self.base.device().device_wait_idle()?;
        Ok(())
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Ok(());
        };

        let in_flight_fence = state.frames.in_flight_fence();

        self.base
            .device()
            .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let image_index = match self.base.device().acquire_next_image_khr(
            state.swapchain.handle(),
            u64::MAX,
            state.frames.image_available(),
            vk::Fence::null(),
        ) {
            Ok((image_index, _)) => image_index,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(e.into()),
        };
        let image_index = image_index as usize;

        state
            .swapchain
            .wait_for_image_fence(image_index, in_flight_fence)?;

        let wait_semaphores = &[state.frames.image_available()];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[state.command_buffers[image_index]];
        let signal_semaphores = &[state.frames.render_finished()];
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

        let swapchains = &[state.swapchain.handle()];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .base
            .device()
            .queue_present_khr(self.base.queues().present_queue, &present_info);
        let swapchain_changed = matches!(
            &result,
            Ok(vk::SuccessCode::SUBOPTIMAL_KHR) | Err(vk::ErrorCode::OUT_OF_DATE_KHR)
        );

        state.frames.next_frame();

        if self.resized || swapchain_changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(e.into());
        }

        Ok(())
    }

    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.wait_idle()?;

        if let Some(mut old_state) = self.state.take() {
            old_state.swapchain_framebuffers.clear();
            self.graphics_command_pool
                .free_command_buffers(&old_state.command_buffers);
        }

        self.state = Some(Self::make_renderer_state(
            &self.base,
            &self.graphics_command_pool,
            window,
        )?);

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
