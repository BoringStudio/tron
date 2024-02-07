use std::mem::MaybeUninit;
use std::sync::{Mutex, MutexGuard};

use anyhow::Result;
use gfx::AsStd140;
use glam::{Mat4, UVec2};

use crate::types::CameraProjection;
use crate::util::Frustum;

pub struct FrameResources {
    descriptor_set_layout: gfx::DescriptorSetLayout,
    descriptor_set: gfx::DescriptorSet,
    camera_data: Mutex<CameraData>,
    buffer: Mutex<UniformBuffer>,
}

impl FrameResources {
    #[tracing::instrument(level = "debug", name = "create_frame_resources", skip_all)]
    pub fn new(device: &gfx::Device) -> Result<Self> {
        // Create descriptor set layout and descriptor set
        let descriptor_set_layout =
            device.create_descriptor_set_layout(gfx::DescriptorSetLayoutInfo {
                bindings: vec![gfx::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: gfx::DescriptorType::UniformBufferDynamic,
                    count: 1,
                    stages: gfx::ShaderStageFlags::ALL,
                    flags: Default::default(),
                }],
                flags: Default::default(),
            })?;
        let descriptor_set = device.create_descriptor_set(gfx::DescriptorSetInfo {
            layout: descriptor_set_layout.clone(),
        })?;

        // Create uniform buffer
        let buffer = UniformBuffer::new(device)?;

        // Bind uniform buffer to descriptor set
        device.update_descriptor_sets(&[gfx::UpdateDescriptorSet {
            set: &descriptor_set,
            writes: &[gfx::DescriptorSetWrite {
                binding: 0,
                element: 0,
                data: gfx::DescriptorSlice::UniformBufferDynamic(&[gfx::BufferRange {
                    buffer: buffer.inner.clone(),
                    offset: 0,
                    size: gfx::align_size(
                        <GpuFrameGlobals as gfx::Std140>::ALIGN_MASK,
                        std::mem::size_of::<GpuFrameGlobals>(),
                    ),
                }]),
            }],
        }]);

        Ok(Self {
            descriptor_set_layout,
            descriptor_set,
            camera_data: Mutex::new(CameraData::default()),
            buffer: Mutex::new(buffer),
        })
    }

    pub fn descriptor_set_layout(&self) -> &gfx::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn descriptor_set(&self) -> &gfx::DescriptorSet {
        &self.descriptor_set
    }

    pub fn set_camera(&self, view: &Mat4, projection: &CameraProjection) {
        let mut camera = self.camera_data.lock().unwrap();
        camera.view = *view;
        camera.projection = *projection;
        camera.updated = true;
    }

    /// Update the uniform buffer and return the byte offset of the updated data
    pub fn flush(&self, args: FlushFrameResources) -> FrameResourcesGuard<'_> {
        const TIME_ROLLOVER: f32 = 3600.0;

        let mut camera_data = self.camera_data.lock().unwrap();

        let mut buffer = self.buffer.lock().unwrap();

        let globals = &mut buffer.globals;

        globals.time = (globals.time + args.delta_time) % TIME_ROLLOVER;
        globals.delta_time = args.delta_time;
        globals.frame_index = args.frame;

        if std::mem::take(&mut camera_data.updated)
            || args.render_resolution != globals.render_resolution
        {
            globals.camera_previous_view = globals.camera_view;
            globals.camera_previous_projection = globals.camera_projection;

            let aspect_ratio = args.render_resolution.x as f32 / args.render_resolution.y as f32;
            globals.render_resolution = args.render_resolution;
            globals.camera_view = camera_data.view;
            globals.camera_projection = camera_data
                .projection
                .compute_projection_matrix(aspect_ratio);
            globals.camera_view_inverse = globals.camera_view.inverse();
            globals.camera_projection_inverse = globals.camera_projection.inverse();
            globals.frustum = Frustum::new(globals.camera_projection * globals.camera_view);

            if !camera_data.initialized {
                globals.camera_previous_view = globals.camera_view;
                globals.camera_previous_projection = globals.camera_projection;
                camera_data.initialized = true;
            }
        }

        buffer.flush();

        FrameResourcesGuard { buffer }
    }
}

pub struct FrameResourcesGuard<'a> {
    buffer: MutexGuard<'a, UniformBuffer>,
}

impl FrameResourcesGuard<'_> {
    pub fn dynamic_offset(&self) -> u32 {
        self.buffer.current_offset()
    }
}

impl std::ops::Deref for FrameResourcesGuard<'_> {
    type Target = FrameGlobals;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer.globals
    }
}

pub struct FlushFrameResources {
    pub render_resolution: UVec2,
    pub delta_time: f32,
    pub frame: u32,
}

struct UniformBuffer {
    globals: FrameGlobals,
    ptr: *mut MaybeUninit<GpuFrameGlobals>,
    slot_len: u32,
    next_frame: usize,
    inner: gfx::Buffer,
}

unsafe impl Send for UniformBuffer {}

impl UniformBuffer {
    fn new(device: &gfx::Device) -> Result<Self> {
        let limits = &device.properties().v1_0.limits;
        let min_offset_align_mask = limits.min_uniform_buffer_offset_alignment as usize - 1;
        let offset_align_mask =
            <GpuFrameGlobals as gfx::Std140>::ALIGN_MASK | min_offset_align_mask;

        // NOTE: Round up to the nearest required alignment
        let slot_len = gfx::align_size(offset_align_mask, std::mem::size_of::<GpuFrameGlobals>());

        // Allocate uniform buffer
        let buffer = device.create_mappable_buffer(
            gfx::BufferInfo {
                align_mask: offset_align_mask,
                size: slot_len * 2,
                usage: gfx::BufferUsage::UNIFORM,
            },
            gfx::MemoryUsage::UPLOAD | gfx::MemoryUsage::FAST_DEVICE_ACCESS,
        )?;

        let ptr = device
            .map_memory(&mut buffer.as_mappable(), 0, slot_len * 2)?
            .as_mut_ptr()
            .cast();

        Ok(Self {
            globals: FrameGlobals::default(),
            ptr,
            slot_len: slot_len as u32,
            next_frame: 1,
            inner: buffer,
        })
    }

    fn current_offset(&self) -> u32 {
        self.slot_len * self.next_frame as u32
    }

    fn flush(&mut self) {
        self.next_frame = 1 - self.next_frame;
        let byte_offset = self.current_offset();

        // SAFETY:
        // - `byte_offset` is always less than `self.slot_len * 2`
        // - `self.ptr` is a valid pointer to mapped memory
        unsafe {
            let ptr = self.ptr.byte_add(byte_offset as usize);
            // TODO: write directly to mapped memory without creating a temporary data on the stack
            *ptr = MaybeUninit::new(self.globals.as_std140());
        }
    }
}

#[derive(AsStd140)]
pub struct FrameGlobals {
    pub frustum: Frustum,
    pub camera_view: Mat4,
    pub camera_projection: Mat4,
    pub camera_view_inverse: Mat4,
    pub camera_projection_inverse: Mat4,
    pub camera_previous_view: Mat4,
    pub camera_previous_projection: Mat4,
    pub render_resolution: UVec2,
    pub time: f32,
    pub delta_time: f32,
    pub frame_index: u32,
}

impl Default for FrameGlobals {
    fn default() -> Self {
        Self {
            frustum: Frustum::IDENTITY,
            camera_view: Mat4::IDENTITY,
            camera_projection: Mat4::IDENTITY,
            camera_view_inverse: Mat4::IDENTITY,
            camera_projection_inverse: Mat4::IDENTITY,
            camera_previous_view: Mat4::IDENTITY,
            camera_previous_projection: Mat4::IDENTITY,
            render_resolution: UVec2::ONE,
            time: 0.0,
            delta_time: f32::EPSILON,
            frame_index: 0,
        }
    }
}

type GpuFrameGlobals = <FrameGlobals as AsStd140>::Output;

struct CameraData {
    view: Mat4,
    projection: CameraProjection,
    initialized: bool,
    updated: bool,
}

impl Default for CameraData {
    fn default() -> Self {
        Self {
            view: Mat4::IDENTITY,
            projection: CameraProjection::default(),
            initialized: false,
            updated: false,
        }
    }
}
