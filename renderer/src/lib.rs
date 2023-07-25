use std::rc::Rc;

use anyhow::Result;
use shared::util::WithDefer;
use vulkanalia::prelude::v1_0::*;
use winit::window::Window;

use self::base::RendererBase;
use self::pipeline::Pipeline;
use self::swapchain::Swapchain;

mod base;
mod pipeline;
mod pipeline_layout;
mod shader_module;
mod swapchain;

pub struct RendererConfig {
    pub app_name: String,
    pub app_version: (u32, u32, u32),
    pub validation_layer_enabled: bool,
}

pub struct Renderer {
    base: Rc<RendererBase>,
    swapchain: Swapchain,
    pipeline: Pipeline,
}

impl Renderer {
    pub unsafe fn new(window: &Window, config: RendererConfig) -> Result<Self> {
        let base = Rc::new(RendererBase::new(window, config)?);

        let mut swapchain = Swapchain::uninit(base.clone());
        swapchain.recreate(window)?;

        let pipeline = Pipeline::new(base.clone())?;

        Ok(Self {
            base,
            swapchain,
            pipeline,
        })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }
}
