use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::Result;
use argh::FromArgs;
use winit::event::*;
use winit::event_loop::EventLoop;
#[cfg(x11_platform)]
use winit::platform::x11::{WindowBuilderExtX11, XWindowType};
use winit::window::WindowBuilder;

use renderer::Renderer;

#[cfg(not(any(target_env = "msvc", miri)))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

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

    /// enable X11-specific popup mode
    #[cfg(x11_platform)]
    #[argh(switch)]
    x11_as_popup: bool,
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
        profiling::puffin::set_scopes_on(self.profiling);

        let app_name = env!("CARGO_BIN_NAME").to_owned();
        let event_loop = EventLoop::new()?;
        let window = {
            let mut builder = WindowBuilder::new();
            builder = builder.with_title(app_name);

            #[cfg(x11_platform)]
            if self.x11_as_popup {
                builder =
                    builder.with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal]);
            }

            builder.build(&event_loop).map(Arc::new)?
        };

        let mut renderer = Renderer::builder(window.clone())
            .app_version((0, 0, 1))
            .validation_layer(self.validation_layer)
            .build()?;

        let is_running = Arc::new(AtomicBool::new(true));

        let rendering_barrier = Arc::new(LoopBarrier::default());
        let renderer_thread = std::thread::Builder::new()
            .name("rendering".to_owned())
            .spawn({
                let is_running = is_running.clone();
                let rendering_barrier = rendering_barrier.clone();
                move || {
                    tracing::debug!("rendering thread started");

                    while is_running.load(Ordering::Acquire) {
                        rendering_barrier.wait();

                        renderer.draw().unwrap();
                    }

                    tracing::debug!("rendering thread stopped");

                    renderer
                }
            })
            .expect("failed to spawn rendering thread");

        let mut minimized = false;
        let handle_event = {
            let rendering_barrier = rendering_barrier.clone();

            move |event, elwt: &winit::event_loop::EventLoopWindowTarget<_>| {
                // elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

                match event {
                    Event::AboutToWait => window.request_redraw(),
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                            rendering_barrier.notify();
                        }
                        WindowEvent::Resized(size) => {
                            minimized = size.width == 0 || size.height == 0;
                        }
                        WindowEvent::CloseRequested => {
                            is_running.store(false, Ordering::Release);
                            rendering_barrier.notify();
                            elwt.exit();
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        };

        tracing::debug!("event loop started");
        event_loop.run(handle_event)?;
        tracing::debug!("event loop stopped");

        renderer_thread.join().unwrap().wait_idle()?;

        Ok(())
    }
}

#[derive(Default)]
struct LoopBarrier {
    state: Mutex<bool>,
    condvar: Condvar,
}

impl LoopBarrier {
    fn wait(&self) {
        let mut state = self.state.lock().unwrap();
        while !*state {
            state = self.condvar.wait(state).unwrap();
        }
        *state = false;
    }

    fn notify(&self) {
        *self.state.lock().unwrap() = true;
        self.condvar.notify_one();
    }
}
