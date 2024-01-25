use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use bevy_ecs::entity::Entity;
use bevy_ecs::system::Resource;
use shared::Embed;
use winit::window::Window;

pub use crate::types::{
    Color, CubeMeshGenerator, Material, MaterialArray, MaterialHandle, MaterialTag, Mesh,
    MeshBuilder, MeshGenerator, MeshHandle, Normal, PlaneMeshGenerator, Position, Sorting,
    SortingOrder, SortingReason, Tangent, VertexAttribute, VertexAttributeData,
    VertexAttributeKind, UV0,
};

use crate::managers::{MaterialManager, MeshManager};
use crate::types::{RawMaterialHandle, RawMeshHandle};
use crate::util::{
    BindlessResources, FrameResources, ResourceHandleAllocator, ScatterCopy, ShaderPreprocessor,
};
use crate::worker::RendererWorker;

pub mod components;
pub mod systems;

mod managers;
mod pipelines;
mod render_passes;
mod types;
mod util;
mod worker;

#[derive(Resource, Default)]
pub struct MainCamera {
    pub entity: Option<Entity>,
}

pub struct RendererBuilder {
    window: Arc<Window>,
    app_version: (u32, u32, u32),
    validation_layer: bool,
    optimize_shaders: bool,
    shaders_debug_info_enabled: bool,
}

impl RendererBuilder {
    pub fn build(self) -> Result<Renderer> {
        let app_version = (0, 0, 1);

        gfx::Graphics::set_init_config(gfx::InstanceConfig {
            app_name: self.window.title().into(),
            app_version,
            validation_layer_enabled: self.validation_layer,
        });

        let graphics = gfx::Graphics::get_or_init()?;
        let (device, queue) = graphics
            .get_physical_devices()?
            .with_required_features(&[
                gfx::DeviceFeature::SurfacePresentation,
                gfx::DeviceFeature::DescriptorBindingUniformBufferUpdateAfterBind,
                gfx::DeviceFeature::DescriptorBindingStorageBufferUpdateAfterBind,
                gfx::DeviceFeature::DescriptorBindingSampledImageUpdateAfterBind,
                gfx::DeviceFeature::DescriptorBindingPartiallyBound,
            ])
            .find_best()?
            .create_logical_device(gfx::SingleQueueQuery::GRAPHICS)?;

        let mesh_manager = MeshManager::new(&device)?;

        let mut shader_preprocessor = ShaderPreprocessor::new();
        shader_preprocessor.set_optimizations_enabled(self.optimize_shaders);
        shader_preprocessor.set_debug_info_enabled(self.shaders_debug_info_enabled);
        for (path, contents) in Shaders::iter() {
            let contents = std::str::from_utf8(contents)
                .with_context(|| anyhow::anyhow!("invalid shader {path}"))?;
            shader_preprocessor.add_file(path, contents)?;
        }

        let scatter_copy = ScatterCopy::new(&device, &shader_preprocessor)?;
        let frame_resources = FrameResources::new(&device)?;
        let bindless_resources = BindlessResources::new(&device)?;

        let mut surface = device.create_surface(self.window.clone())?;
        surface.configure()?;

        let state = Arc::new(RendererState {
            is_running: AtomicBool::new(true),
            window_resized: AtomicBool::new(true),
            worker_barrier: LoopBarrier::default(),
            instructions: InstructionQueue::default(),
            mesh_manager,
            synced_managers: Default::default(),
            handles: Default::default(),
            scatter_copy,
            frame_resources,
            bindless_resources,
            shader_preprocessor,
            window: self.window,
            queue,
            device,
        });

        let mut worker = RendererWorker::new(state.clone(), surface)?;

        let worker_thread = std::thread::spawn({
            let state = state.clone();

            move || {
                tracing::debug!("rendering thread started");

                let state = state.as_ref();
                while state.is_running.load(Ordering::Acquire) {
                    state.worker_barrier.wait();
                    worker.draw().unwrap();
                }

                tracing::debug!("rendering thread stopped");
            }
        });

        Ok(Renderer {
            state,
            worker_thread: Some(worker_thread),
        })
    }

    pub fn app_version(mut self, app_version: (u32, u32, u32)) -> Self {
        self.app_version = app_version;
        self
    }

    pub fn validation_layer(mut self, validation_layer: bool) -> Self {
        self.validation_layer = validation_layer;
        self
    }

    pub fn optimize_shaders(mut self, optimize_shaders: bool) -> Self {
        self.optimize_shaders = optimize_shaders;
        self
    }

    pub fn shaders_debug_info_enabled(mut self, shaders_debug_info_enabled: bool) -> Self {
        self.shaders_debug_info_enabled = shaders_debug_info_enabled;
        self
    }
}

pub struct Renderer {
    state: Arc<RendererState>,
    worker_thread: Option<std::thread::JoinHandle<()>>,
}

impl Renderer {
    pub fn builder(window: Arc<Window>) -> RendererBuilder {
        RendererBuilder {
            window,
            app_version: (0, 0, 1),
            validation_layer: false,
            optimize_shaders: true,
            shaders_debug_info_enabled: false,
        }
    }

    pub fn state(&self) -> &Arc<RendererState> {
        &self.state
    }

    pub fn cleanup(&mut self) -> Result<()> {
        if let Some(worker_thread) = self.worker_thread.take() {
            self.state.set_running(false);
            worker_thread.join().unwrap();
            self.state.device.wait_idle()?;
        }
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            tracing::error!("failed to cleanup renderer: {e:?}");
        }
    }
}

pub struct RendererState {
    is_running: AtomicBool,
    worker_barrier: LoopBarrier,
    instructions: InstructionQueue,

    mesh_manager: MeshManager,
    synced_managers: Mutex<RendererStateSyncedManagers>,
    handles: RendererStateHandles,

    scatter_copy: ScatterCopy,
    frame_resources: FrameResources,
    bindless_resources: BindlessResources,
    shader_preprocessor: ShaderPreprocessor,

    window: Arc<Window>,
    window_resized: AtomicBool,
    queue: gfx::Queue,

    // NOTE: device must be dropped last
    device: gfx::Device,
}

impl RendererState {
    pub fn set_running(&self, is_running: bool) {
        self.is_running.store(is_running, Ordering::Release);
        self.worker_barrier.notify();
    }

    pub fn notify_draw(&self) {
        self.worker_barrier.notify();
    }

    pub fn notify_resized(&self) {
        self.window_resized.store(true, Ordering::Release);
    }

    pub fn add_mesh(self: &Arc<Self>, mesh: &Mesh) -> Result<MeshHandle> {
        let mesh = self.mesh_manager.upload_mesh(&self.queue, mesh)?;

        let state = Arc::downgrade(self);
        let handle = self
            .handles
            .mesh_handle_allocator
            .alloc(Arc::new(move |handle| {
                if let Some(state) = state.upgrade() {
                    state.instructions.send(Instruction::DeleteMesh { handle });
                }
            }));

        self.mesh_manager.insert(handle.raw(), mesh);
        Ok(handle)
    }

    pub fn add_material<M: Material>(self: &Arc<Self>, material: M) -> MaterialHandle {
        let state = Arc::downgrade(self);
        let handle = self
            .handles
            .material_handle_allocator
            .alloc(Arc::new(move |handle| {
                if let Some(state) = state.upgrade() {
                    state
                        .instructions
                        .send(Instruction::DeleteMaterial { handle });
                }
            }));

        self.instructions.send(Instruction::AddMaterial {
            handle: handle.raw(),
            on_add: Box::new(move |manager, device, handle| {
                manager.insert(device, handle, material)
            }),
        });
        handle
    }

    pub fn update_material<M: Material>(self: &Arc<Self>, handle: &MaterialHandle, material: M) {
        self.instructions.send(Instruction::UpdateMaterial {
            handle: handle.raw(),
            on_update: Box::new(move |manager, handle| manager.update(handle, material)),
        });
    }

    pub(crate) fn eval_instructions(&self, encoder: &mut gfx::Encoder) -> Result<()> {
        self.instructions.swap();

        self.bindless_resources.flush_retired();

        let mut instructions = self.instructions.consumer.lock().unwrap();

        let mut synced_managers = self.synced_managers.lock().unwrap();
        let synced_managers = &mut *synced_managers;

        for instruction in instructions.drain(..) {
            match instruction {
                Instruction::DeleteMesh { handle } => {
                    self.handles.mesh_handle_allocator.dealloc(handle);
                    self.mesh_manager.remove(handle);
                }
                Instruction::AddMaterial { handle, on_add } => {
                    on_add(&mut synced_managers.material_manager, &self.device, handle)?;
                }
                Instruction::UpdateMaterial { handle, on_update } => {
                    on_update(&mut synced_managers.material_manager, handle);
                }
                Instruction::DeleteMaterial { handle } => {
                    self.handles.material_handle_allocator.dealloc(handle);
                    synced_managers.material_manager.remove(handle);
                }
            }
        }

        // TEMP
        synced_managers.material_manager.flush::<DebugMaterial>(
            &self.device,
            encoder,
            &self.scatter_copy,
        )?;

        Ok(())
    }
}

#[derive(Default)]
struct RendererStateSyncedManagers {
    material_manager: MaterialManager,
}

#[derive(Default)]
struct RendererStateHandles {
    mesh_handle_allocator: ResourceHandleAllocator<Mesh>,
    material_handle_allocator: ResourceHandleAllocator<MaterialTag>,
}

#[derive(Default)]
struct InstructionQueue {
    consumer: Mutex<Vec<Instruction>>,
    producer: Mutex<Vec<Instruction>>,
}

impl InstructionQueue {
    fn swap(&self) {
        let mut consumer = self.consumer.lock().unwrap();
        let mut producer = self.producer.lock().unwrap();
        std::mem::swap(&mut *consumer, &mut *producer);
    }

    fn send(&self, instruction: Instruction) {
        self.producer.lock().unwrap().push(instruction);
    }
}

enum Instruction {
    DeleteMesh {
        handle: RawMeshHandle,
    },
    AddMaterial {
        handle: RawMaterialHandle,
        on_add: Box<FnOnAddMaterial>,
    },
    UpdateMaterial {
        handle: RawMaterialHandle,
        on_update: Box<FnOnUpdateMaterial>,
    },
    DeleteMaterial {
        handle: RawMaterialHandle,
    },
}

type FnOnAddMaterial = dyn FnOnce(
        &mut MaterialManager,
        &gfx::Device,
        RawMaterialHandle,
    ) -> Result<(), gfx::OutOfDeviceMemory>
    + Send
    + Sync;
type FnOnUpdateMaterial = dyn FnOnce(&mut MaterialManager, RawMaterialHandle) + Send + Sync;

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

shared::embed!(
    Shaders("../../assets/shaders") = [
        "math/color.glsl",
        "math/const.glsl",
        "uniforms/bindless.glsl",
        "uniforms/globals.glsl",
        "scatter_copy.comp",
        "opaque_mesh.vert",
        "opaque_mesh.frag"
    ]
);

// TEMP!
#[derive(Debug, Clone, Copy)]
pub struct DebugMaterial {
    pub color: glam::Vec3,
}

impl Material for DebugMaterial {
    type ShaderDataType = <glam::Vec3 as gfx::AsStd430>::Output;

    fn required_attributes() -> impl MaterialArray<VertexAttributeKind> {
        [VertexAttributeKind::Position]
    }
    fn supported_attributes() -> impl MaterialArray<VertexAttributeKind> {
        [
            VertexAttributeKind::Position,
            VertexAttributeKind::Normal,
            VertexAttributeKind::Tangent,
            VertexAttributeKind::UV0,
            VertexAttributeKind::Color,
        ]
    }

    fn key(&self) -> u64 {
        0
    }

    fn sorting(&self) -> Sorting {
        Sorting::OPAQUE
    }

    fn shader_data(&self) -> Self::ShaderDataType {
        gfx::AsStd430::as_std430(&self.color)
    }
}
