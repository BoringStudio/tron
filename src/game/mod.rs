use std::hash::Hash;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use anyhow::Result;

use self::input_map::*;
use crate::renderer::managers::MeshManager;
use crate::renderer::pipelines::geometry_pipeline::InstanceDescription;
use crate::renderer::types::{MeshHandle, Texture};
use crate::renderer::Renderer;

mod gltf_scene;
mod input_map;

pub struct Game {
    renderer: Renderer,
    mesh_manager: MeshManager,
    input_map: InputMap<InputAction>,
    started_at: Instant,

    resource_id: AtomicUsize,

    is_running: bool,
}

impl Game {
    pub async fn new(window: &winit::window::Window) -> Result<Self> {
        let renderer = Renderer::new(window).await?;
        let mut mesh_manager = MeshManager::new(renderer.device());

        let mut input_map = InputMap::default();
        input_map.set_action_key(InputAction::Close, KeyCode::Escape)?;
        input_map.set_action_key(InputAction::MoveForward, KeyCode::W)?;
        input_map.set_action_key(InputAction::MoveBackward, KeyCode::S)?;
        input_map.set_action_key(InputAction::MoveLeft, KeyCode::A)?;
        input_map.set_action_key(InputAction::MoveRight, KeyCode::D)?;

        let started_at = Instant::now();

        let resource_id = AtomicUsize::new(0);

        let mut encoder = renderer.encode_commands("init");
        let meshes = gltf_scene::load(
            &mut encoder,
            &mut mesh_manager,
            &resource_id,
            include_bytes!("./res/bike.glb"),
        )?;

        println!("{meshes:#?}");

        let texture = Texture::from_bytes(
            encoder.device,
            encoder.queue,
            include_bytes!("./res/texture.png"),
            "texture",
        )?;

        Ok(Self {
            renderer,
            mesh_manager,
            input_map,
            started_at,
            resource_id,
            is_running: true,
        })
    }

    pub fn is_running(&self) -> bool {
        self.is_running
    }

    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::WindowEvent;

        let action = match event {
            WindowEvent::CloseRequested => return self.close(),
            WindowEvent::KeyboardInput { input, .. } => self.input_map.handle_input(input),
            _ => None,
        };

        if let Some(InputState::Pressed(InputAction::Close)) = action {
            self.close();
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.renderer.resize(size);
    }

    pub fn render(&mut self) {
        let time = self.started_at.elapsed().as_secs_f32();
        self.renderer.render(time);
    }

    fn close(&mut self) {
        self.is_running = false;
    }
}

#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
enum InputAction {
    Close,
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
}

impl nohash_hasher::IsEnabled for InputAction {}

struct TestObject {
    mesh: MeshHandle,
    texture: Texture,
    descr: InstanceDescription,
}
