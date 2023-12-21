#![allow(clippy::redundant_closure_call)]

use std::sync::Arc;

use anyhow::{Context, Result};
use argh::FromArgs;
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

        let event_loop = EventLoop::new()?;
        let window = WindowBuilder::new()
            .with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal])
            .with_title(&app_name)
            .build(&event_loop)
            .map(Arc::new)?;

        let graphics = gfx::Graphics::get_or_init()?;
        let physical = graphics.get_physical_devices()?.find_best()?;
        let (device, _queue) = physical.create_device(
            &[gfx::DeviceFeature::SurfacePresentation],
            gfx::SingleQueueQuery::GRAPHICS,
        )?;

        let mut surface = device.create_surface(window.clone())?;
        surface.configure()?;

        tracing::debug!("starting event loop");

        let mut minimized = false;
        event_loop.run(move |event, elwt| {
            // elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

            match event {
                Event::AboutToWait => window.request_redraw(),
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                        (|| -> anyhow::Result<()> {
                            // let mut image = surface.aquire_image()?;

                            // let [wait, signal] = image.wait_signal();

                            Ok(())
                        })()
                        .unwrap()
                    }
                    WindowEvent::Resized(size) => {
                        if size.width == 0 || size.height == 0 {
                            minimized = true;
                        } else {
                            minimized = false;
                            // TODO: update window?
                        }
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
