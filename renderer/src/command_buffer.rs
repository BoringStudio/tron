use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::base::RendererBase;

pub struct CommandPool {
    base: Rc<RendererBase>,
    handle: vk::CommandPool,
}

impl CommandPool {
    pub unsafe fn new_graphics_command_pool(base: Rc<RendererBase>) -> Result<Self> {
        let queue_family_index = base.physical_device().graphics_queue_family_idx;
        Self::new(
            base,
            vk::CommandPoolCreateFlags::empty(),
            queue_family_index,
        )
    }

    pub unsafe fn new_transient_command_pool(base: Rc<RendererBase>) -> Result<Self> {
        let queue_family_index = base.physical_device().graphics_queue_family_idx;
        Self::new(
            base,
            vk::CommandPoolCreateFlags::TRANSIENT,
            queue_family_index,
        )
    }

    pub unsafe fn new(
        base: Rc<RendererBase>,
        flags: vk::CommandPoolCreateFlags,
        queue_family_index: u32,
    ) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::builder()
            .flags(flags)
            .queue_family_index(queue_family_index);

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

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.base.device().destroy_command_pool(self.handle, None);
        }
    }
}
