use anyhow::Result;
use winit::event::*;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::x11::{WindowBuilderExtX11, XWindowType};
use winit::window::WindowBuilder;

use game::Game;

mod game;
mod renderer;
mod util;

pub async fn run() -> Result<()> {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal])
        .build(&event_loop)?;

    let mut game = Game::new(&window).await?;

    event_loop.run(move |event, _, control_flow| {
        if !game.is_running() {
            *control_flow = ControlFlow::Exit;
            return;
        }

        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::WindowEvent {
                event:
                    WindowEvent::Resized(mut size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut mut size,
                        ..
                    },
                ..
            } => {
                size.width = size.width.max(1);
                size.height = size.height.max(1);
                game.resize(size);
            }
            Event::WindowEvent { event, .. } => game.handle_event(&event),
            Event::RedrawRequested(_) => game.render(),
            _ => {}
        }
    });
}

fn main() -> Result<()> {
    pollster::block_on(run())
}
