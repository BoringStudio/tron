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
        unsafe {
            self.base.device().destroy_semaphore(self.handle, None);
        }
    }
}
