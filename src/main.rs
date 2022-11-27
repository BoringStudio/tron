use anyhow::Result;
use winit::event::*;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::unix::{WindowBuilderExtUnix, XWindowType};
use winit::window::WindowBuilder;

use self::renderer::Renderer;

mod renderer;
mod scene;
mod util;

pub async fn run() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal])
        .build(&event_loop)?;

    let mut renderer = Renderer::new(&window).await?;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => renderer.resize(size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                renderer.resize(*new_inner_size)
            }
            _ => {}
        },
        Event::RedrawRequested(window_id) if window_id == window.id() => match renderer.render() {
            Err(wgpu::SurfaceError::Lost) => renderer.resize(renderer.size()),
            Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
            _ => {}
        },
        Event::MainEventsCleared => window.request_redraw(),
        _ => {}
    });
}

fn main() {
    if let Err(e) = pollster::block_on(run()) {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}
