use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::base::RendererBase;

pub struct PipelineLayout {
    base: Rc<RendererBase>,
    handle: vk::PipelineLayout,
}

impl PipelineLayout {
    pub unsafe fn new(base: Rc<RendererBase>) -> Result<Self> {
        let info = vk::PipelineLayoutCreateInfo::builder();

        // TODO: add builder

        let handle = base.device().create_pipeline_layout(&info, None)?;

        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.base
                .device()
                .destroy_pipeline_layout(self.handle, None)
        }
    }
}
