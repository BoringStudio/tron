use std::rc::Rc;

use anyhow::Result;
use shared::util::WithDefer;
use vulkanalia::prelude::v1_0::*;

use crate::base::RendererBase;
use crate::command_buffer::CommandPool;

pub trait BufferCreateInfoExt {
    unsafe fn make_buffer(
        &self,
        base: Rc<RendererBase>,
        flags: gpu_alloc::UsageFlags,
    ) -> Result<Buffer>;
}

impl BufferCreateInfoExt for vk::BufferCreateInfo {
    unsafe fn make_buffer(
        &self,
        base: Rc<RendererBase>,
        flags: gpu_alloc::UsageFlags,
    ) -> Result<Buffer> {
        Buffer::new(base, self, flags)
    }
}

pub struct Buffer {
    base: Rc<RendererBase>,
    handle: vk::Buffer,
    memory: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
}

impl Buffer {
    pub unsafe fn new(
        base: Rc<RendererBase>,
        create_info: &vk::BufferCreateInfo,
        flags: gpu_alloc::UsageFlags,
    ) -> Result<Self> {
        let device = base.device();
        let buffer = device
            .create_buffer(create_info, None)?
            .with_defer(|buffer| device.destroy_buffer(buffer, None));

        let req = device.get_buffer_memory_requirements(*buffer);

        let memory = base.allocator().borrow_mut().alloc(
            base.memory_device(),
            gpu_alloc::Request {
                size: req.size,
                align_mask: req.alignment,
                usage: flags,
                memory_types: req.memory_type_bits,
            },
        )?;

        device.bind_buffer_memory(*buffer, *memory.memory(), memory.offset())?;

        Ok(Self {
            handle: buffer.disarm(),
            base,
            memory: Some(memory),
        })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    pub unsafe fn write_bytes(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        let memory = self.memory.as_mut().unwrap();
        memory.write_bytes(self.base.memory_device(), offset, data)?;
        Ok(())
    }

    pub unsafe fn copy_from(
        &mut self,
        other: &Buffer,
        size: u64,
        command_pool: &CommandPool,
    ) -> Result<()> {
        let device = self.base.device();
        let queue = self.base.queues().graphics_queue;

        let info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool.handle())
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device.begin_command_buffer(command_buffer, &info)?;

        let regions = vk::BufferCopy::builder().size(size);
        device.cmd_copy_buffer(command_buffer, other.handle, self.handle, &[regions]);

        device.end_command_buffer(command_buffer)?;

        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder().command_buffers(command_buffers);
        device.queue_submit(queue, &[info], vk::Fence::null())?;

        // TODO: use fences instead
        device.queue_wait_idle(queue)?;

        device.free_command_buffers(command_pool.handle(), command_buffers);

        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if let Some(block) = self.memory.take() {
                self.base
                    .allocator()
                    .borrow_mut()
                    .dealloc(self.base.memory_device(), block);
            }

            self.base.device().destroy_buffer(self.handle, None);
        }
    }
}
