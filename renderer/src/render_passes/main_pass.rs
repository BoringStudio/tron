use anyhow::Result;
use gfx::MakeImageView;

use crate::render_passes::Pass;

pub struct MainPassInput {
    pub max_image_count: usize,
    pub target: gfx::Image,
}

#[derive(Default)]
pub struct MainPass {
    render_pass: Option<gfx::RenderPass>,
    framebuffers: Vec<gfx::Framebuffer>,
}

impl MainPass {
    #[tracing::instrument(level = "debug", name = "create_main_pass", skip_all)]
    fn get_or_init_framebuffer(
        &mut self,
        device: &gfx::Device,
        input: &MainPassInput,
    ) -> Result<&gfx::Framebuffer> {
        'compat: {
            let Some(render_pass) = &self.render_pass else {
                break 'compat;
            };

            let target_attachment = &render_pass.info().attachments[0];
            if target_attachment.format != input.target.info().format
                || target_attachment.samples != input.target.info().samples
            {
                break 'compat;
            }

            //
            let target_image_info = input.target.info();
            match self.framebuffers.iter().position(|fb| {
                let attachment = fb.info().attachments[0].info();
                attachment.image == input.target
                    && attachment.range
                        == gfx::ImageSubresourceRange::new(
                            target_image_info.format.aspect_flags(),
                            0..1,
                            0..1,
                        )
            }) {
                Some(index) => {
                    let framebuffer = self.framebuffers.remove(index);
                    self.framebuffers.push(framebuffer);
                }
                None => {
                    let framebuffer = device.create_framebuffer(gfx::FramebufferInfo {
                        render_pass: render_pass.clone(),
                        attachments: vec![
                            input.target.make_image_view(device)?,
                            make_depth_attachment(device, &input.target)?,
                        ],
                        extent: target_image_info.extent.into(),
                    })?;

                    let to_remove =
                        (self.framebuffers.len() + 1).saturating_sub(input.max_image_count);
                    if to_remove > 0 {
                        self.framebuffers.drain(0..to_remove);
                    }
                    self.framebuffers.push(framebuffer);
                }
            };

            return Ok(self.framebuffers.last().unwrap());
        };

        self.recreate_render_pass(device, input)
    }

    fn recreate_render_pass(
        &mut self,
        device: &gfx::Device,
        input: &MainPassInput,
    ) -> Result<&gfx::Framebuffer> {
        let target_image_info = input.target.info();

        let attachments = vec![
            gfx::AttachmentInfo {
                format: target_image_info.format,
                samples: target_image_info.samples,
                load_op: gfx::LoadOp::Clear(()),
                store_op: gfx::StoreOp::Store,
                initial_layout: None,
                final_layout: gfx::ImageLayout::ColorAttachmentOptimal,
            },
            gfx::AttachmentInfo {
                format: gfx::Format::D32Sfloat,
                samples: gfx::Samples::_1,
                load_op: gfx::LoadOp::Clear(()),
                store_op: gfx::StoreOp::DontCare,
                initial_layout: None,
                final_layout: gfx::ImageLayout::DepthStencilAttachmentOptimal,
            },
        ];

        let subpasses = vec![gfx::Subpass {
            colors: vec![(0, gfx::ImageLayout::ColorAttachmentOptimal)],
            depth: Some((1, gfx::ImageLayout::DepthStencilAttachmentOptimal)),
        }];

        let dependencies = vec![gfx::SubpassDependency {
            src: None,
            src_stages: gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | gfx::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst: Some(0),
            dst_stages: gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | gfx::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        }];

        let render_pass =
            self.render_pass
                .insert(device.create_render_pass(gfx::RenderPassInfo {
                    attachments,
                    subpasses,
                    dependencies,
                })?);

        //
        let framebuffer_info = match self.framebuffers.iter().find(|fb| {
            let attachment = fb.info().attachments[0].info();
            attachment.image == input.target
                && attachment.range
                    == gfx::ImageSubresourceRange::new(
                        target_image_info.format.aspect_flags(),
                        0..1,
                        0..1,
                    )
        }) {
            Some(fb) => gfx::FramebufferInfo {
                render_pass: render_pass.clone(),
                attachments: fb.info().attachments.clone(),
                extent: fb.info().extent,
            },
            None => gfx::FramebufferInfo {
                render_pass: render_pass.clone(),
                attachments: vec![
                    input.target.make_image_view(device)?,
                    make_depth_attachment(device, &input.target)?,
                ],
                extent: target_image_info.extent.into(),
            },
        };
        self.framebuffers.clear();
        self.framebuffers
            .push(device.create_framebuffer(framebuffer_info)?);

        Ok(self.framebuffers.last().unwrap())
    }
}

fn make_depth_attachment(
    device: &gfx::Device,
    target: &gfx::Image,
) -> Result<gfx::ImageView, gfx::OutOfDeviceMemory> {
    device
        .create_image(gfx::ImageInfo {
            extent: target.info().extent,
            format: gfx::Format::D32Sfloat,
            mip_levels: 1,
            samples: gfx::Samples::_1,
            array_layers: 1,
            usage: gfx::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        })?
        .make_image_view(device)
}

impl Pass for MainPass {
    type Input = MainPassInput;

    fn begin_render_pass<'a, 'b>(
        &'b mut self,
        input: &Self::Input,
        device: &gfx::Device,
        encoder: &'a mut gfx::Encoder,
    ) -> Result<gfx::RenderPassEncoder<'a, 'b>> {
        let framebuffer = self.get_or_init_framebuffer(device, input)?;
        Ok(encoder.with_framebuffer(
            framebuffer,
            &[
                gfx::ClearColor(0.02, 0.02, 0.02, 1.0).into(),
                gfx::ClearDepth(1.0).into(),
            ],
        ))
    }
}
