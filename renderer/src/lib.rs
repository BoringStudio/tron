use std::rc::Rc;

use anyhow::Result;
use shared::util::WithDefer;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::window::Window;

use self::base::RendererBase;
use self::command_buffer::GraphicsCommandPool;
use self::pipeline::Pipeline;
use self::swapchain::Swapchain;
use self::sync::Semaphore;

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
    render_finished_semaphore: Semaphore,
    image_available_semaphore: Semaphore,
    swapchain_framebuffers: Vec<Framebuffer>,
    graphics_command_pool: GraphicsCommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl Renderer {
    pub unsafe fn new(window: &Window, config: RendererConfig) -> Result<Self> {
        let base = Rc::new(RendererBase::new(window, config)?);

        let mut swapchain = Swapchain::uninit(base.clone());
        swapchain.recreate(window)?;

        let pipeline = Pipeline::new(base.clone())?;

        let render_finished_semaphore = Semaphore::new(base.clone())?;
        let image_available_semaphore = Semaphore::new(base.clone())?;

        let swapchain_framebuffers = swapchain
            .image_views()
            .iter()
            .map(|image_view| {
                Framebuffer::new(
                    base.clone(),
                    pipeline.render_pass().handle(),
                    *image_view,
                    swapchain.extent(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

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
            render_finished_semaphore,
            image_available_semaphore,
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
        let (image_index, _) = self.base.device().acquire_next_image_khr(
            self.swapchain.handle(),
            u64::MAX,
            self.image_available_semaphore.handle(),
            vk::Fence::null(),
        )?;

        let wait_semaphores = &[self.image_available_semaphore.handle()];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.render_finished_semaphore.handle()];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.base.device().queue_submit(
            self.base.queues().graphics_queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )?;

        let swapchains = &[self.swapchain.handle()];
        let image_indices = &[image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.base
            .device()
            .queue_present_khr(self.base.queues().present_queue, &present_info)?;

        Ok(())
    }
}

struct Framebuffer {
    base: Rc<RendererBase>,
    handle: vk::Framebuffer,
}

impl Framebuffer {
    pub unsafe fn new(
        base: Rc<RendererBase>,
        render_pass: vk::RenderPass,
        image_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&image_view))
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let handle = base.device().create_framebuffer(&info, None)?;

        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::Framebuffer {
        self.handle
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.base.device().destroy_framebuffer(self.handle, None);
        }
    }
}
