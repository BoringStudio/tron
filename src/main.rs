extern crate nalgebra_glm as glm;

use anyhow::Result;
use winit::event::*;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::unix::{WindowBuilderExtUnix, XWindowType};
use winit::window::WindowBuilder;

use self::window_state::WindowState;

mod camera;
mod geometry_pipeline;
mod mesh;
mod screen_pipeline;
mod texture;
mod window_state;

pub async fn run() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal])
        .build(&event_loop)?;

    let mut state = WindowState::new(&window).await?;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => state.resize(*new_inner_size),
            _ => {}
        },
        Event::RedrawRequested(window_id) if window_id == window.id() => match state.render() {
            Err(wgpu::SurfaceError::Lost) => state.resize(state.size()),
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
