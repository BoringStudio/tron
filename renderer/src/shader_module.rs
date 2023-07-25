use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::base::RendererBase;

pub struct ShaderModule {
    base: Rc<RendererBase>,
    handle: vk::ShaderModule,
}

impl ShaderModule {
    pub unsafe fn new(base: Rc<RendererBase>, code: &[u32]) -> Result<Self> {
        let info = vk::ShaderModuleCreateInfo::builder()
            .code_size(code.len() * 4)
            .code(code);

        let handle = base.device().create_shader_module(&info, None)?;
        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.base.device().destroy_shader_module(self.handle, None);
        }
    }
}
