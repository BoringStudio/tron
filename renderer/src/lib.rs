use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use shared::Embed;
use vulkanalia::vk;
use winit::window::Window;

pub use crate::types::{
    BoundingSphere, Camera, CameraProjection, Color, CubeMeshGenerator, Frustum, Material,
    MaterialArray, MaterialHandle, MaterialTag, Mesh, MeshBuilder, MeshGenerator, MeshHandle,
    Normal, Plane, PlaneMeshGenerator, Position, Sorting, SortingOrder, SortingReason, Tangent,
    VertexAttribute, VertexAttributeData, VertexAttributeKind, UV0,
};

use crate::managers::{MaterialManager, MeshManager};
use crate::resource_handle::ResourceHandleAllocator;
use crate::types::{RawMaterialHandle, RawMeshHandle};
use crate::util::{ScatterCopy, ShaderPreprocessor};
use crate::worker::{RendererWorker, RendererWorkerCallbacks, RendererWorkerConfig};

mod managers;
mod pipelines;
mod render_passes;
mod resource_handle;
mod types;
mod util;
mod worker;

pub struct RendererBuilder {
    window: Arc<Window>,
    app_version: (u32, u32, u32),
    validation_layer: bool,
    frames_in_flight: NonZeroUsize,
    optimize_shaders: bool,
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
        let physical = graphics.get_physical_devices()?.find_best()?;
        let (device, queue) = physical.create_device(
            &[gfx::DeviceFeature::SurfacePresentation],
            gfx::SingleQueueQuery::GRAPHICS,
        )?;

        let mesh_manager = MeshManager::new(&device)?;

        let mut shader_preprocessor = ShaderPreprocessor::new();
        shader_preprocessor.set_optimizations_enabled(self.optimize_shaders);
        for (path, contents) in Shaders::iter() {
            let contents = std::str::from_utf8(contents)
                .with_context(|| anyhow::anyhow!("invalid shader {path}"))?;
            shader_preprocessor.add_file(path, contents)?;
        }

        let scatter_copy = ScatterCopy::new(&device, &shader_preprocessor)?;

        let mut surface = device.create_surface(self.window.clone())?;
        surface.configure()?;

        let state = Arc::new(RendererState {
            is_running: AtomicBool::new(true),
            worker_barrier: LoopBarrier::default(),
            instructions: InstructionQueue::default(),
            mesh_manager,
            synced_managers: Default::default(),
            handles: Default::default(),
            scatter_copy,
            shader_preprocessor,
            queue,
            device,
        });

        let mut worker = RendererWorker::new(
            state.clone(),
            RendererWorkerConfig {
                frames_in_flight: self.frames_in_flight,
            },
            Box::new(WorkerCallbacks {
                window: self.window.clone(),
            }),
            surface,
        )?;

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

    pub fn frames_in_flight(mut self, frames_in_flight: usize) -> Self {
        self.frames_in_flight = frames_in_flight.try_into().unwrap();
        self
    }

    pub fn optimize_shaders(mut self, optimize_shaders: bool) -> Self {
        self.optimize_shaders = optimize_shaders;
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
            frames_in_flight: NonZeroUsize::new(2).unwrap(),
            optimize_shaders: true,
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
    shader_preprocessor: ShaderPreprocessor,
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

struct WorkerCallbacks {
    window: Arc<Window>,
}

impl RendererWorkerCallbacks for WorkerCallbacks {
    fn before_present(&self) {
        self.window.pre_present_notify();
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

shared::embed!(
    Shaders("../../assets/shaders") = [
        "math/color.glsl",
        "math/const.glsl",
        "scatter_copy.comp",
        "triangle.vert",
        "triangle.frag"
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
