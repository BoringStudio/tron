use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::base::RendererBase;

pub struct Semaphore {
    base: Rc<RendererBase>,
    handle: vk::Semaphore,
}

impl Semaphore {
    pub unsafe fn new(base: Rc<RendererBase>) -> Result<Self> {
        let info = vk::SemaphoreCreateInfo::builder();

        let handle = base.device().create_semaphore(&info, None)?;

        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.base.device().destroy_semaphore(self.handle, None) }
    }
}

pub struct Fence {
    base: Rc<RendererBase>,
    handle: vk::Fence,
}

impl Fence {
    pub unsafe fn new(base: Rc<RendererBase>, signaled: bool) -> Result<Self> {
        let mut info = vk::FenceCreateInfo::builder();
        if signaled {
            info.flags = vk::FenceCreateFlags::SIGNALED;
        }

        let handle = base.device().create_fence(&info, None)?;

        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::Fence {
        self.handle
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self.base.device().destroy_fence(self.handle, None) }
    }
}
