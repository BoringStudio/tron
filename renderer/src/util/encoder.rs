use anyhow::Result;

pub trait EncoderExt {
    fn with_render_pass<'a, 'b, P>(
        &'a mut self,
        pass: &'b mut P,
        input: &P::Input,
        device: &gfx::Device,
    ) -> Result<gfx::RenderPassEncoder<'a, 'b>>
    where
        P: RenderPass;
}

impl EncoderExt for gfx::Encoder {
    fn with_render_pass<'a, 'b, P>(
        &'a mut self,
        pass: &'b mut P,
        input: &P::Input,
        device: &gfx::Device,
    ) -> Result<gfx::RenderPassEncoder<'a, 'b>>
    where
        P: RenderPass,
    {
        pass.begin_render_pass(input, device, self)
    }
}

pub trait RenderPass {
    type Input;

    fn begin_render_pass<'a, 'b>(
        &'b mut self,
        input: &Self::Input,
        device: &gfx::Device,
        encoder: &'a mut gfx::Encoder,
    ) -> Result<gfx::RenderPassEncoder<'a, 'b>>;
}

pub trait RenderPassEncoderExt {
    fn bind_cached_graphics_pipeline(
        &mut self,
        pipeline: &mut CachedGraphicsPipeline,
        device: &gfx::Device,
    ) -> Result<()>;
}

impl RenderPassEncoderExt for gfx::RenderPassEncoder<'_, '_> {
    fn bind_cached_graphics_pipeline(
        &mut self,
        pipeline: &mut CachedGraphicsPipeline,
        device: &gfx::Device,
    ) -> Result<()> {
        let mut set_viewport = false;
        let mut set_scissor = false;

        if let Some(rasterizer) = &pipeline.descr().rasterizer {
            set_viewport = rasterizer.viewport.is_dynamic();
            set_scissor = rasterizer.scissor.is_dynamic();
        }

        if set_viewport {
            let mut viewport: gfx::Viewport = self.framebuffer().info().extent.into();
            viewport.y.offset = viewport.y.size;
            viewport.y.size = -viewport.y.size;
            self.set_viewport(&viewport);
        }
        if set_scissor {
            let scissor = self.framebuffer().info().extent.into();
            self.set_scissor(&scissor);
        }

        let pipeline = pipeline.prepare(device, self.render_pass(), 0)?;
        self.bind_graphics_pipeline(pipeline);
        Ok(())
    }
}

pub struct CachedGraphicsPipeline {
    descr: gfx::GraphicsPipelineDescr,
    cached: Option<gfx::GraphicsPipeline>,
}

impl CachedGraphicsPipeline {
    pub fn new(descr: gfx::GraphicsPipelineDescr) -> Self {
        Self {
            cached: None,
            descr,
        }
    }

    pub fn descr(&self) -> &gfx::GraphicsPipelineDescr {
        &self.descr
    }

    pub fn prepare(
        &mut self,
        device: &gfx::Device,
        render_pass: &gfx::RenderPass,
        subpass: u32,
    ) -> Result<&gfx::GraphicsPipeline> {
        if let Some(pipeline) = &mut self.cached {
            let info = pipeline.info();

            let compatible =
                &info.rendering.render_pass == render_pass && info.rendering.subpass == subpass;

            if !compatible || info.descr != self.descr {
                self.cached = None;
            }
        }

        Ok(match &mut self.cached {
            Some(pipeline) => pipeline,
            cached => cached.get_or_insert(device.create_graphics_pipeline(
                gfx::GraphicsPipelineInfo {
                    descr: self.descr.clone(),
                    rendering: gfx::GraphicsPipelineRenderingInfo {
                        render_pass: render_pass.clone(),
                        subpass,
                    },
                },
            )?),
        })
    }
}
