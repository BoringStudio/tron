use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::window::Window;

use crate::base::{RendererBase, SwapchainSupport};

pub struct GraphicsCommandPool {
    base: Rc<RendererBase>,
    handle: vk::CommandPool,
}

impl GraphicsCommandPool {
    pub unsafe fn new(base: Rc<RendererBase>) -> Result<Self> {
        // TODO: use TRANSIENT flag to record a lot of unique buffers from the single pool
        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::empty())
            .queue_family_index(base.physical_device().graphics_queue_family_idx);

        let handle = base.device().create_command_pool(&info, None)?;

        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::CommandPool {
        self.handle
    }

    pub fn allocate_command_buffers(&self, n: usize) -> Result<Vec<vk::CommandBuffer>> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(n as u32);

        unsafe {
            self.base
                .device()
                .allocate_command_buffers(&info)
                .map_err(From::from)
        }
    }

    pub fn free_command_buffers(&mut self, buffers: &[vk::CommandBuffer]) {
        unsafe {
            self.base
                .device()
                .free_command_buffers(self.handle, buffers)
        }
    }
}

impl Drop for GraphicsCommandPool {
    fn drop(&mut self) {
        unsafe {
            self.base.device().destroy_command_pool(self.handle, None);
        }
    }
}
