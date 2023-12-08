use std::rc::Rc;

use anyhow::Result;
use argh::FromArgs;
use winit::event::*;
use winit::event_loop::EventLoop;
use winit::platform::x11::{WindowBuilderExtX11, XWindowType};
use winit::window::WindowBuilder;

use renderer::*;

fn main() -> Result<()> {
    let app: App = argh::from_env();
    app.run()
}

/// Vulkan rendering experiments
#[derive(FromArgs)]
struct App {
    /// don't set up vulkan validation layer
    #[argh(switch)]
    disable_validation_layer: bool,
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

        let event_loop = EventLoop::new()?;
        let window = WindowBuilder::new()
            .with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal])
            .with_title(&app_name)
            .build(&event_loop)
            .map(Rc::new)?;

        let mut renderer = unsafe {
            Renderer::new(
                window.clone(),
                RendererConfig {
                    app_name,
                    app_version,
                    validation_layer_enabled: !self.disable_validation_layer,
                },
            )?
        };

        let mut minimized = false;
        event_loop.run(move |event, elwt| match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                    unsafe { renderer.render() }.unwrap();
                }
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        renderer.mark_resized();
                    }
                }
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe { renderer.wait_idle() }.unwrap();
                }
                _ => {}
            },
            _ => {}
        })?;

        Ok(())
    }
}
