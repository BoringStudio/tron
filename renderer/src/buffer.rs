use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::base::RendererBase;

pub trait BufferCreateInfoExt {
    unsafe fn make_buffer(&self, base: Rc<RendererBase>) -> Result<Buffer>;
}

impl BufferCreateInfoExt for vk::BufferCreateInfo {
    unsafe fn make_buffer(&self, base: Rc<RendererBase>) -> Result<Buffer> {
        Buffer::new(base, self)
    }
}

pub struct Buffer {
    base: Rc<RendererBase>,
    handle: vk::Buffer,
    memory: Option<gpu_alloc::MemoryBlock<vk::DeviceMemory>>,
}

impl Buffer {
    pub unsafe fn new(base: Rc<RendererBase>, create_info: &vk::BufferCreateInfo) -> Result<Self> {
        let device = base.device();
        let handle = device.create_buffer(create_info, None)?;

        let req = device.get_buffer_memory_requirements(handle);
        let memory = base.allocator().borrow_mut().alloc(
            base.memory_device(),
            gpu_alloc::Request {
                size: req.size,
                align_mask: req.alignment,
                usage: gpu_alloc::UsageFlags::HOST_ACCESS,
                memory_types: !0,
            },
        )?;

        device.bind_buffer_memory(handle, *memory.memory(), 0)?;

        Ok(Self {
            base,
            handle,
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
