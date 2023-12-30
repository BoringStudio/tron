use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

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
    /// enable profiling server
    #[argh(switch)]
    profiling: bool,
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

        let _puffin_server = if self.profiling {
            let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
            let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
            tracing::info!(server_addr, "started profiling server");
            Some(puffin_server)
        } else {
            None
        };
        puffin::set_scopes_on(self.profiling);

        let app_name = env!("CARGO_BIN_NAME").to_owned();
        let event_loop = EventLoop::new()?;
        let window = WindowBuilder::new()
            .with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal])
            .with_title(&app_name)
            .build(&event_loop)
            .map(Arc::new)?;

        let mut renderer = Renderer::new(window.clone(), self.validation_layer)?;

        let is_running = Arc::new(AtomicBool::new(true));
        let should_render = Arc::new((Mutex::new(false), Condvar::new()));
        let renderer_thread = std::thread::spawn({
            let is_running = is_running.clone();
            let should_render = should_render.clone();
            move || {
                let (should_render, should_render_notify) = &*should_render;

                tracing::debug!("rendering thread started");

                while is_running.load(Ordering::Relaxed) {
                    {
                        let mut should_render = should_render.lock().unwrap();
                        while !*should_render {
                            should_render = should_render_notify.wait(should_render).unwrap();
                        }
                    }

                    renderer.draw().unwrap();
                }

                tracing::debug!("rendering thread stopped");

                renderer
            }
        });

        tracing::debug!("starting event loop");

        let mut minimized = false;
        event_loop.run(move |event, elwt| {
            // elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

            match event {
                Event::AboutToWait => window.request_redraw(),
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                        let (should_render, should_render_notify) = &*should_render;
                        *should_render.lock().unwrap() = true;
                        should_render_notify.notify_one();
                    }
                    WindowEvent::Resized(size) => {
                        minimized = size.width == 0 || size.height == 0;
                    }
                    WindowEvent::CloseRequested => {
                        is_running.store(false, Ordering::Relaxed);
                        elwt.exit();
                    }
                    _ => {}
                },
                _ => {}
            }
        })?;

        tracing::debug!("event loop stopped");

        let renderer = renderer_thread.join().unwrap();
        renderer.device.wait_idle()
    }
}

struct Renderer {
    window: Arc<winit::window::Window>,
    device: gfx::Device,
    queue: gfx::Queue,
    surface: gfx::Surface,
    pass: MainPass,
    fences: Box<[gfx::Fence]>,
    fence_index: usize,
    non_optimal_count: usize,
}

impl Renderer {
    fn new(window: Arc<winit::window::Window>, validation_layer_enabled: bool) -> Result<Self> {
        let app_version = (0, 0, 1);

        gfx::Graphics::set_init_config(gfx::InstanceConfig {
            app_name: window.title().into(),
            app_version,
            validation_layer_enabled,
        });
        let pass = MainPass::default();

        let graphics = gfx::Graphics::get_or_init()?;
        let physical = graphics.get_physical_devices()?.find_best()?;
        let (device, queue) = physical.create_device(
            &[gfx::DeviceFeature::SurfacePresentation],
            gfx::SingleQueueQuery::GRAPHICS,
        )?;

        let mut surface = device.create_surface(window.clone())?;
        surface.configure()?;

        let fences = (0..FRAMES_INFLIGHT)
            .map(|_| device.create_fence())
            .collect::<Result<Box<[_]>>>()?;

        Ok(Self {
            window,
            device,
            queue,
            surface,
            pass,
            fences,
            fence_index: 0,
            non_optimal_count: 0,
        })
    }

    fn draw(&mut self) -> Result<()> {
        let fence_count = self.fences.len();
        let fence = &mut self.fences[self.fence_index];
        self.fence_index = (self.fence_index + 1) % fence_count;
        if !fence.state().is_unsignalled() {
            puffin::profile_scope!("idle");
            self.device.wait_fences(&mut [fence], true)?;
            self.device.reset_fences(&mut [fence])?;
        }

        puffin::profile_scope!("frame");

        let mut surface_image = {
            puffin::profile_scope!("aquire_image");
            self.surface.aquire_image()?
        };

        let mut encoder = self.queue.create_encoder()?;

        {
            let _render_pass = encoder.with_framebuffer(
                self.pass.get_or_init_framebuffer(
                    &self.device,
                    &MainPassInput {
                        max_image_count: surface_image.total_image_count(),
                        target: surface_image.image().clone(),
                    },
                )?,
                &[gfx::ClearColor(0.02, 0.02, 0.02, 1.0).into()],
            );
        }

        let [wait, signal] = surface_image.wait_signal();

        {
            puffin::profile_scope!("queue_submit");
            self.queue.submit(
                &mut [(gfx::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, wait)],
                Some(encoder.finish()?),
                &mut [signal],
                Some(fence),
            )?;
        }

        let mut is_optimal = surface_image.is_optimal();
        {
            puffin::profile_scope!("queue_present");

            self.window.pre_present_notify();
            match self.queue.present(surface_image)? {
                gfx::PresentStatus::Ok => {}
                gfx::PresentStatus::Suboptimal => is_optimal = false,
                gfx::PresentStatus::OutOfDate => {
                    is_optimal = false;
                    self.non_optimal_count += NON_OPTIMAL_LIMIT;
                }
            }
        }

        self.non_optimal_count += !is_optimal as usize;
        if self.non_optimal_count >= NON_OPTIMAL_LIMIT {
            puffin::profile_scope!("recreate_swapchain");

            // Wait for the device to be idle before recreating the swapchain.
            self.device.wait_idle()?;

            self.surface.update()?;

            self.non_optimal_count = 0;
        }

        puffin::GlobalProfiler::lock().new_frame();
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

const NON_OPTIMAL_LIMIT: usize = 100;
const FRAMES_INFLIGHT: usize = 2;
