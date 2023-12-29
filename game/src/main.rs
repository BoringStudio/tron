#![allow(clippy::redundant_closure_call)]

use std::sync::Arc;

use anyhow::{Context, Result};
use argh::FromArgs;
use gfx::MakeImageView;
use vulkanalia::vk;
use winit::event::*;
use winit::event_loop::EventLoop;
use winit::platform::x11::{WindowBuilderExtX11, XWindowType};
use winit::window::WindowBuilder;

fn main() -> Result<()> {
    let app: App = argh::from_env();
    app.run()
}

/// Vulkan rendering experiments
#[derive(FromArgs)]
struct App {
    /// don't set up vulkan validation layer
    #[argh(switch)]
    validation_layer: bool,
}

impl App {
    pub fn run(self) -> Result<()> {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::builder()
                    .with_default_directive(tracing::Level::INFO.into())
                    .from_env_lossy(),
            )
            .init();

        let app_name = env!("CARGO_BIN_NAME").to_owned();
        let app_version = (0, 0, 1);

        gfx::Graphics::set_init_config(gfx::InstanceConfig {
            app_name: app_name.clone().into(),
            app_version,
            validation_layer_enabled: self.validation_layer,
        });
        let mut pass = MainPass::default();

        let event_loop = EventLoop::new()?;
        let window = WindowBuilder::new()
            .with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal])
            .with_title(&app_name)
            .build(&event_loop)
            .map(Arc::new)?;

        let graphics = gfx::Graphics::get_or_init()?;
        let physical = graphics.get_physical_devices()?.find_best()?;
        let (device, mut queue) = physical.create_device(
            &[gfx::DeviceFeature::SurfacePresentation],
            gfx::SingleQueueQuery::GRAPHICS,
        )?;

        let mut surface = device.create_surface(window.clone())?;
        surface.configure()?;

        let mut fences = (0..FRAMES_INFLIGHT)
            .map(|_| device.create_fence())
            .collect::<Result<Box<[_]>>>()?;
        let fence_count = fences.len();
        let mut fence_index = 0;

        tracing::debug!("starting event loop");

        let mut minimized = false;
        let mut resized = false;
        let mut non_optimal_count = 0;
        event_loop.run(move |event, elwt| {
            // elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

            match event {
                Event::AboutToWait => window.request_redraw(),
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                        (|| -> anyhow::Result<()> {
                            let fence = &mut fences[fence_index];
                            fence_index = (fence_index + 1) % fence_count;
                            if !fence.state().is_unsignalled() {
                                device.wait_fences(&mut [fence], true)?;
                                device.reset_fences(&mut [fence])?;
                            }

                            let mut surface_image = surface.aquire_image()?;

                            let mut encoder = queue.create_encoder()?;

                            {
                                let _render_pass = encoder.with_framebuffer(
                                    pass.get_or_init_framebuffer(
                                        &device,
                                        &MainPassInput {
                                            max_image_count: surface_image.total_image_count(),
                                            target: surface_image.image().clone(),
                                        },
                                    )?,
                                    &[gfx::ClearColor(0.02, 0.02, 0.02, 1.0).into()],
                                );
                            }

                            let [wait, signal] = surface_image.wait_signal();

                            queue.submit(
                                &mut [(gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, wait)],
                                Some(encoder.finish()?),
                                &mut [signal],
                                Some(fence),
                            )?;

                            let mut is_optimal = surface_image.is_optimal();
                            match queue.present(surface_image)? {
                                gfx::PresentStatus::Ok => {}
                                gfx::PresentStatus::Suboptimal => is_optimal = false,
                                gfx::PresentStatus::OutOfDate => {
                                    is_optimal = false;
                                    non_optimal_count += NON_OPTIMAL_LIMIT;
                                }
                            }

                            non_optimal_count += !is_optimal as u32;
                            if resized || non_optimal_count >= NON_OPTIMAL_LIMIT {
                                // Wait for the device to be idle before recreating the swapchain.
                                device.wait_idle()?;

                                surface.update()?;

                                resized = false;
                                non_optimal_count = 0;
                            }

                            Ok(())
                        })()
                        .unwrap()
                    }
                    WindowEvent::Resized(size) => {
                        minimized = size.width == 0 || size.height == 0;
                        resized = true;
                    }
                    WindowEvent::CloseRequested => {
                        device.wait_idle().unwrap();
                        elwt.exit();
                    }
                    _ => {}
                },
                _ => {}
            }
        })?;

        tracing::debug!("event loop stopped");
        Ok(())
    }
}

struct MainPassInput {
    max_image_count: usize,
    target: gfx::Image,
}

#[derive(Default)]
struct MainPass {
    render_pass: Option<gfx::RenderPass>,
    framebuffers: Vec<gfx::Framebuffer>,
}

impl MainPass {
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
                        attachments: vec![input.target.make_image_view(device)?],
                        extent: target_image_info.extent.into(),
                    })?;

                    if self.framebuffers.len() + 1 > input.max_image_count {
                        let to_remove = self.framebuffers.len() + 1 - input.max_image_count;
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

        let attachments = vec![gfx::AttachmentInfo {
            format: target_image_info.format,
            samples: target_image_info.samples,
            load_op: gfx::LoadOp::Clear(()),
            store_op: gfx::StoreOp::Store,
            initial_layout: None,
            final_layout: gfx::ImageLayout::Present,
        }];

        let subpasses = vec![gfx::Subpass {
            colors: vec![(0, gfx::ImageLayout::ColorAttachmentOptimal)],
            depth: None,
        }];

        let dependencies = vec![gfx::SubpassDependency {
            src: None,
            src_stages: gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst: Some(0),
            dst_stages: gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
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
                attachments: vec![input.target.make_image_view(device)?],
                extent: target_image_info.extent.into(),
            },
        };
        self.framebuffers.clear();
        self.framebuffers
            .push(device.create_framebuffer(framebuffer_info)?);

        Ok(self.framebuffers.last().unwrap())
    }
}

trait PhysicalDevicesExt {
    fn find_best(self) -> Result<gfx::PhysicalDevice>;
}

impl PhysicalDevicesExt for Vec<gfx::PhysicalDevice> {
    fn find_best(mut self) -> Result<gfx::PhysicalDevice> {
        let mut result = None;

        for (index, physical_device) in self.iter().enumerate() {
            let properties = physical_device.properties();

            let mut score = 0usize;
            match properties.v1_0.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => score += 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => score += 100,
                vk::PhysicalDeviceType::CPU => score += 10,
                vk::PhysicalDeviceType::VIRTUAL_GPU => score += 1,
                _ => continue,
            }

            tracing::info!(
                name = %properties.v1_0.device_name,
                ty = ?properties.v1_0.device_type,
                "found physical device",
            );

            match &result {
                Some((_index, best_score)) if *best_score >= score => continue,
                _ => result = Some((index, score)),
            }
        }

        let (index, _) = result.context("no suitable physical device found")?;
        Ok(self.swap_remove(index))
    }
}

const NON_OPTIMAL_LIMIT: u32 = 100u32;
const FRAMES_INFLIGHT: usize = 2;
