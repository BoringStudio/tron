use std::mem::MaybeUninit;
use std::sync::Mutex;

use anyhow::Result;
use gfx::AsStd140;
use glam::{Mat4, Vec2};

use crate::types::CameraProjection;

pub struct FrameResources {
    descriptor_set_layout: gfx::DescriptorSetLayout,
    descriptor_set: gfx::DescriptorSet,
    buffer: Mutex<UniformBuffer>,
    globals: Mutex<Globals>,
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
                    // NOTE: size is one slot, not the whole buffer due to dynamic offsets usage
                    size: buffer.slot_len as _,
                }]),
            }],
        }]);

        Ok(Self {
            descriptor_set_layout,
            descriptor_set,
            buffer: Mutex::new(buffer),
            globals: Mutex::new(Globals::default()),
        })
    }

    pub fn descriptor_set_layout(&self) -> &gfx::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn descriptor_set(&self) -> &gfx::DescriptorSet {
        &self.descriptor_set
    }

    pub fn set_render_resolution(&self, width: u32, height: u32) {
        let mut globals = self.globals.lock().unwrap();
        globals.render_resolution = glam::vec2(width as f32, height as f32);
    }

    pub fn set_camera(&self, view: &Mat4, projection: &CameraProjection) {
        let mut globals = self.globals.lock().unwrap();

        globals.camera_previous_view = globals.camera_view;
        globals.camera_previous_projection = globals.camera_projection;

        let aspect_ratio = globals.render_resolution.x / globals.render_resolution.y;
        globals.camera_view = *view;
        globals.camera_projection = projection.compute_projection_matrix(aspect_ratio);
        globals.camera_view_inverse = view.inverse();
        globals.camera_projection_inverse = globals.camera_projection.inverse();
    }

    /// Update the uniform buffer and return the byte offset of the updated data
    pub fn flush(&self, delta_time: f32, frame: u32) -> u32 {
        const TIME_ROLLOVER: f32 = 3600.0;

        let mut globals = self.globals.lock().unwrap();

        globals.time = (globals.time + delta_time) % TIME_ROLLOVER;
        globals.delta_time = delta_time;
        globals.frame_index = frame;
        self.buffer.lock().unwrap().write(&globals)
    }
}

struct UniformBuffer {
    ptr: *mut MaybeUninit<GpuGlobals>,
    slot_len: u32,
    odd_frame: bool,
    inner: gfx::Buffer,
}

unsafe impl Send for UniformBuffer {}

impl UniformBuffer {
    fn new(device: &gfx::Device) -> Result<Self> {
        let limits = &device.properties().v1_0.limits;
        let min_offset_align_mask = limits.min_uniform_buffer_offset_alignment as usize - 1;
        let offset_align_mask = <GpuGlobals as gfx::Std140>::ALIGN_MASK | min_offset_align_mask;

        // NOTE: Round up to the nearest required alignment
        let slot_len = gfx::align_size(offset_align_mask, std::mem::size_of::<GpuGlobals>());

        // Allocate uniform buffer
        let mut buffer = device.create_mappable_buffer(
            gfx::BufferInfo {
                align_mask: offset_align_mask,
                size: slot_len * 2,
                usage: gfx::BufferUsage::UNIFORM,
            },
            gfx::MemoryUsage::UPLOAD | gfx::MemoryUsage::FAST_DEVICE_ACCESS,
        )?;

        let ptr = device
            .map_memory(&mut buffer, 0, (slot_len * 2) as usize)?
            .as_mut_ptr()
            .cast();

        Ok(Self {
            ptr,
            slot_len: slot_len as u32,
            odd_frame: false,
            inner: buffer.freeze(),
        })
    }

    fn write(&mut self, globals: &Globals) -> u32 {
        let byte_offset = self.slot_len * self.odd_frame as u32;
        // SAFETY:
        // - `byte_offset` is always less than `self.slot_len * 2`
        // - `self.ptr` is a valid pointer to mapped memory
        unsafe {
            let ptr = self.ptr.byte_add(byte_offset as usize);
            // TODO: write directly to mapped memory without creating a temporary data on the stack
            (*ptr).write(globals.as_std140());
        }

        self.odd_frame = !self.odd_frame;
        byte_offset
    }
}

#[derive(AsStd140)]
struct Globals {
    camera_view: Mat4,
    camera_projection: Mat4,
    camera_view_inverse: Mat4,
    camera_projection_inverse: Mat4,
    camera_previous_view: Mat4,
    camera_previous_projection: Mat4,
    render_resolution: Vec2,
    time: f32,
    delta_time: f32,
    frame_index: u32,
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            camera_view: Mat4::IDENTITY,
            camera_projection: Mat4::IDENTITY,
            camera_view_inverse: Mat4::IDENTITY,
            camera_projection_inverse: Mat4::IDENTITY,
            camera_previous_view: Mat4::IDENTITY,
            camera_previous_projection: Mat4::IDENTITY,
            render_resolution: Vec2::ONE,
            time: 0.0,
            delta_time: f32::EPSILON,
            frame_index: 0,
        }
    }
}

type GpuGlobals = <Globals as AsStd140>::Output;
