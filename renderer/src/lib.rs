use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use glam::Mat4;
use shared::Embed;
use winit::window::Window;

pub use self::render_graph::materials;
pub use crate::types::{
    CameraProjection, Color, CubeMeshGenerator, DynamicObjectHandle, MaterialInstance,
    MaterialInstanceHandle, MaterialInstanceTag, Mesh, MeshBuilder, MeshGenerator, MeshHandle,
    Normal, PlaneMeshGenerator, Position, Sorting, SortingOrder, SortingReason, StaticObjectHandle,
    Tangent, VertexAttribute, VertexAttributeData, VertexAttributeKind, UV0,
};

use crate::managers::{MaterialManager, MeshManager, ObjectManager, TimeManager};
use crate::types::{RawMaterialInstanceHandle, RawMeshHandle, RawStaticObjectHandle};
use crate::util::{
    BindlessResources, FrameResources, FreelistHandleAllocator, HandleAllocator, HandleData,
    HandleDeleter, MultiBufferArena, RawResourceHandle, ScatterCopy, ShaderPreprocessor,
    SimpleHandleAllocator,
};
use crate::worker::RendererWorker;

use self::types::{DynamicObjectTag, ObjectData, RawDynamicObjectHandle, StaticObjectTag};

mod managers;
mod render_graph;
mod types;
mod util;
mod worker;

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
                gfx::DeviceFeature::ShaderStorageBufferNonUniformIndexing,
                gfx::DeviceFeature::DescriptorBindingUniformBufferUpdateAfterBind,
                gfx::DeviceFeature::DescriptorBindingStorageBufferUpdateAfterBind,
                gfx::DeviceFeature::DescriptorBindingSampledImageUpdateAfterBind,
                gfx::DeviceFeature::DescriptorBindingPartiallyBound,
            ])
            .find_best()?
            .create_logical_device(gfx::SingleQueueQuery::GRAPHICS)?;

        let mut shader_preprocessor = ShaderPreprocessor::new();
        shader_preprocessor.set_optimizations_enabled(self.optimize_shaders);
        shader_preprocessor.set_debug_info_enabled(self.shaders_debug_info_enabled);
        for (path, contents) in Shaders::iter() {
            let contents = std::str::from_utf8(contents)
                .with_context(|| anyhow::anyhow!("invalid shader {path}"))?;
            shader_preprocessor.add_file(path, contents)?;
        }

        let frame_resources = FrameResources::new(&device)?;
        let bindless_resources = BindlessResources::new(&device)?;
        let scatter_copy = ScatterCopy::new(&device, &shader_preprocessor)?;
        let multi_buffer_arena = MultiBufferArena::new(&device);

        let mesh_manager = MeshManager::new(&device, &bindless_resources)?;

        let mut surface = device.create_surface(self.window.clone())?;
        surface.configure()?;

        let state = Arc::new(RendererState {
            is_running: AtomicBool::new(true),
            worker_barrier: LoopBarrier::default(),
            instructions: InstructionQueue::default(),
            mesh_manager,
            synced_managers: Default::default(),
            handles: Default::default(),
            frame_resources,
            bindless_resources,
            multi_buffer_arena,
            scatter_copy,
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

    frame_resources: FrameResources,
    bindless_resources: BindlessResources,
    multi_buffer_arena: MultiBufferArena,
    shader_preprocessor: ShaderPreprocessor,
    scatter_copy: ScatterCopy,

    window: Arc<Window>,
    queue: gfx::Queue,

    // NOTE: device must be dropped last
    device: gfx::Device,
}

impl RendererState {
    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    pub fn set_running(&self, is_running: bool) {
        self.is_running.store(is_running, Ordering::Release);
        self.worker_barrier.notify();
    }

    pub fn notify_draw(&self) {
        self.worker_barrier.notify();
    }

    pub fn update_camera(&self, view: &Mat4, projection: &CameraProjection) {
        self.frame_resources.set_camera(view, projection);
    }

    pub fn add_mesh(self: &Arc<Self>, mesh: &Mesh) -> Result<MeshHandle> {
        let mesh = self.mesh_manager.upload_mesh(&self.queue, mesh)?;

        let state = Arc::downgrade(self);
        let handle = self
            .handles
            .mesh_handle_allocator
            .alloc(Arc::new(InstructedHandleDeleter(state)));

        self.mesh_manager.add(handle.raw(), mesh);
        Ok(handle)
    }

    pub fn add_material_instance<M: MaterialInstance>(
        self: &Arc<Self>,
        material: M,
    ) -> MaterialInstanceHandle {
        let state = Arc::downgrade(self);
        let handle = self
            .handles
            .material_handle_allocator
            .alloc(Arc::new(InstructedHandleDeleter(state)));

        self.instructions.send(Instruction::AddMaterialInstance {
            handle: handle.raw(),
            on_add: Box::new(move |manager, handle| {
                manager.insert_material_instance(handle, material)
            }),
        });
        handle
    }

    pub fn update_material<M: MaterialInstance>(
        self: &Arc<Self>,
        handle: &MaterialInstanceHandle,
        material: M,
    ) {
        self.instructions.send(Instruction::UpdateMaterial {
            handle: handle.raw(),
            on_update: Box::new(move |manager, handle| manager.update(handle, material)),
        });
    }

    pub fn add_static_object(
        self: &Arc<Self>,
        mesh_handle: MeshHandle,
        material_handle: MaterialInstanceHandle,
        global_transform: &Mat4,
    ) -> StaticObjectHandle {
        let state = Arc::downgrade(self);
        let handle = self
            .handles
            .static_object_handle_allocator
            .alloc(Arc::new(InstructedHandleDeleter(state)));

        self.instructions.send(Instruction::AddStaticObject {
            handle: handle.raw(),
            object: Box::new(ObjectData {
                mesh: mesh_handle,
                material: material_handle,
                global_transform: *global_transform,
            }),
        });
        handle
    }

    pub fn add_dynamic_object(
        self: &Arc<Self>,
        mesh_handle: MeshHandle,
        material_handle: MaterialInstanceHandle,
        global_transform: &Mat4,
    ) -> DynamicObjectHandle {
        let state = Arc::downgrade(self);
        let handle = self
            .handles
            .dynamic_object_handle_allocator
            .alloc(Arc::new(InstructedHandleDeleter(state)));

        self.instructions.send(Instruction::AddDynamicObject {
            handle: handle.raw(),
            object: Box::new(ObjectData {
                mesh: mesh_handle,
                material: material_handle,
                global_transform: *global_transform,
            }),
        });
        handle
    }

    pub fn update_static_object(self: &Arc<Self>, handle: &StaticObjectHandle, transform: Mat4) {
        self.instructions.send(Instruction::UpdateStaticObject {
            handle: handle.raw(),
            transform: Box::new(transform),
        });
    }

    pub fn update_dynamic_object(
        self: &Arc<Self>,
        handle: &DynamicObjectHandle,
        transform: Mat4,
        teleport: bool,
    ) {
        self.instructions.send(Instruction::UpdateDynamicObject {
            handle: handle.raw(),
            transform: Box::new(transform),
            teleport,
        });
    }

    pub fn finish_fixed_update(self: &Arc<Self>, updated_at: Instant, duration: Duration) {
        self.instructions.send(Instruction::FinishFixedUpdate {
            updated_at,
            duration,
        });
    }

    #[tracing::instrument(level = "debug", name = "eval_instructions", skip_all)]
    pub(crate) fn eval_instructions<'a>(
        &'a self,
        encoder: &mut gfx::PrimaryEncoder,
    ) -> Result<MutexGuard<'a, RendererStateSyncedManagers>> {
        self.instructions.swap();

        self.bindless_resources.flush_retired();

        let mut instructions = self.instructions.consumer.lock().unwrap();

        let mut synced_managers = self.synced_managers.lock().unwrap();

        let mut mesh_manager_data = None;

        for instruction in instructions.drain(..) {
            let synced_managers = &mut *synced_managers;
            match instruction {
                Instruction::RemoveMesh { handle } => {
                    tracing::trace!(?handle, "remove_mesh");
                    self.handles.mesh_handle_allocator.dealloc(handle);
                    self.mesh_manager.remove(handle);
                }
                Instruction::AddMaterialInstance { handle, on_add } => {
                    tracing::trace!(?handle, "add_material");
                    on_add(&mut synced_managers.material_manager, handle);
                }
                Instruction::UpdateMaterial { handle, on_update } => {
                    tracing::trace!(?handle, "update_material");
                    on_update(&mut synced_managers.material_manager, handle);
                }
                Instruction::RemoveMaterial { handle } => {
                    tracing::trace!(?handle, "remove_material");
                    self.handles.material_handle_allocator.dealloc(handle);
                    synced_managers.material_manager.remove(handle);
                }
                Instruction::AddStaticObject { handle, object } => {
                    tracing::trace!(?handle, "add_static_object");
                    let inner_meshes =
                        mesh_manager_data.get_or_insert_with(|| self.mesh_manager.lock_data());

                    synced_managers.object_manager.add_static_object(
                        handle,
                        object,
                        inner_meshes,
                        &mut synced_managers.material_manager,
                    );
                }
                Instruction::AddDynamicObject { handle, object } => {
                    tracing::trace!(?handle, "add_dynamic_object");
                    let inner_meshes =
                        mesh_manager_data.get_or_insert_with(|| self.mesh_manager.lock_data());

                    synced_managers.object_manager.add_dynamic_object(
                        handle,
                        object,
                        inner_meshes,
                        &mut synced_managers.material_manager,
                    );
                }
                Instruction::UpdateStaticObject { handle, transform } => {
                    tracing::trace!(?handle, "update_static_object");
                    synced_managers
                        .object_manager
                        .update_static_object(handle, transform.as_ref());
                }
                Instruction::UpdateDynamicObject {
                    handle,
                    transform,
                    teleport,
                } => {
                    tracing::trace!(?handle, "update_dynamic_object");
                    synced_managers.object_manager.update_dynamic_object(
                        handle,
                        transform.as_ref(),
                        teleport,
                    );
                }
                Instruction::RemoveStaticObject { handle } => {
                    tracing::trace!(?handle, "remove_static_object");
                    self.handles.static_object_handle_allocator.dealloc(handle);
                    synced_managers.object_manager.remove_static_object(handle);
                }
                Instruction::RemoveDynamicObject { handle } => {
                    tracing::trace!(?handle, "remove_dynamic_object");
                    self.handles.dynamic_object_handle_allocator.dealloc(handle);
                    synced_managers.object_manager.remove_dynamic_object(handle);
                }
                Instruction::FinishFixedUpdate {
                    updated_at,
                    duration,
                } => {
                    tracing::trace!(?updated_at, ?duration, "finish_fixed_update");

                    synced_managers
                        .object_manager
                        .finalize_dynamic_object_transforms();

                    synced_managers
                        .time_manager
                        .updated_fixed_time(updated_at, duration);
                }
            }
        }

        synced_managers.object_manager.flush_static_objects(
            &self.device,
            encoder,
            &self.scatter_copy,
            &self.bindless_resources,
            &self.multi_buffer_arena,
        )?;

        synced_managers.material_manager.flush(
            &self.device,
            encoder,
            &self.scatter_copy,
            &self.bindless_resources,
            &self.multi_buffer_arena,
        )?;

        if let Some(secondary) = self
            .mesh_manager
            .drain(&self.device, &self.bindless_resources)
        {
            // NOTE: MeshManager registry must not be touched
            encoder.execute_commands(std::iter::once(secondary.finish()?));
        }

        self.multi_buffer_arena.flush(&self.bindless_resources);

        Ok(synced_managers)
    }
}

#[derive(Default)]
struct RendererStateSyncedManagers {
    material_manager: MaterialManager,
    object_manager: ObjectManager,
    time_manager: TimeManager,
}

#[derive(Default)]
struct RendererStateHandles {
    mesh_handle_allocator: FreelistHandleAllocator<Mesh>,
    material_handle_allocator: SimpleHandleAllocator<MaterialInstanceTag>,
    static_object_handle_allocator: SimpleHandleAllocator<StaticObjectTag>,
    dynamic_object_handle_allocator: SimpleHandleAllocator<DynamicObjectTag>,
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
    RemoveMesh {
        handle: RawMeshHandle,
    },
    AddMaterialInstance {
        handle: RawMaterialInstanceHandle,
        on_add: Box<FnOnAddMaterial>,
    },
    UpdateMaterial {
        handle: RawMaterialInstanceHandle,
        on_update: Box<FnOnUpdateMaterial>,
    },
    RemoveMaterial {
        handle: RawMaterialInstanceHandle,
    },
    AddStaticObject {
        handle: RawStaticObjectHandle,
        object: Box<ObjectData>,
    },
    AddDynamicObject {
        handle: RawDynamicObjectHandle,
        object: Box<ObjectData>,
    },
    UpdateStaticObject {
        handle: RawStaticObjectHandle,
        transform: Box<Mat4>,
    },
    UpdateDynamicObject {
        handle: RawDynamicObjectHandle,
        transform: Box<Mat4>,
        teleport: bool,
    },
    RemoveStaticObject {
        handle: RawStaticObjectHandle,
    },
    RemoveDynamicObject {
        handle: RawDynamicObjectHandle,
    },
    FinishFixedUpdate {
        updated_at: Instant,
        duration: Duration,
    },
}

type FnOnAddMaterial = dyn FnOnce(&mut MaterialManager, RawMaterialInstanceHandle) + Send + Sync;
type FnOnUpdateMaterial = dyn FnOnce(&mut MaterialManager, RawMaterialInstanceHandle) + Send + Sync;

trait IntoRemoveInstruction {
    fn into_remove_instruction(self) -> Instruction;
}

impl IntoRemoveInstruction for RawMeshHandle {
    #[inline]
    fn into_remove_instruction(self) -> Instruction {
        Instruction::RemoveMesh { handle: self }
    }
}

impl IntoRemoveInstruction for RawMaterialInstanceHandle {
    #[inline]
    fn into_remove_instruction(self) -> Instruction {
        Instruction::RemoveMaterial { handle: self }
    }
}

impl IntoRemoveInstruction for RawStaticObjectHandle {
    #[inline]
    fn into_remove_instruction(self) -> Instruction {
        Instruction::RemoveStaticObject { handle: self }
    }
}

impl IntoRemoveInstruction for RawDynamicObjectHandle {
    #[inline]
    fn into_remove_instruction(self) -> Instruction {
        Instruction::RemoveDynamicObject { handle: self }
    }
}

#[doc(hidden)]
pub struct InstructedHandleDeleter(Weak<RendererState>);

impl<T> HandleDeleter<T> for InstructedHandleDeleter
where
    RawResourceHandle<T>: IntoRemoveInstruction,
{
    fn delete(&self, handle: RawResourceHandle<T>) {
        if let Some(state) = self.0.upgrade() {
            state.instructions.send(handle.into_remove_instruction());
        }
    }
}

impl HandleData for Mesh {
    type Deleter = InstructedHandleDeleter;
}

impl HandleData for MaterialInstanceTag {
    type Deleter = InstructedHandleDeleter;
}

impl HandleData for StaticObjectTag {
    type Deleter = InstructedHandleDeleter;
}

impl HandleData for DynamicObjectTag {
    type Deleter = InstructedHandleDeleter;
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

shared::embed!(
    Shaders("../../assets/shaders") = [
        "math/color.glsl",
        "math/const.glsl",
        "math/frustum.glsl",
        "math/sphere.glsl",
        "uniforms/bindless.glsl",
        "uniforms/globals.glsl",
        "uniforms/object.glsl",
        "scatter_copy.comp",
        "opaque_mesh.vert",
        "opaque_mesh.frag"
    ]
);
