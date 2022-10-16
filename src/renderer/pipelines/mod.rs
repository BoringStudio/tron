use wgpu::util::DeviceExt;

use super::types::Camera;

pub use self::geometry_pipeline::GeometryPipeline;
pub use self::screen_pipeline::ScreenPipeline;
pub use self::sky_pipeline::SkyPipeline;

pub mod geometry_pipeline;
pub mod screen_pipeline;
pub mod sky_pipeline;

pub struct BasePipelineBuffer {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl BasePipelineBuffer {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("base_pipeline_buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bind_group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
            }],
        });

        Self {
            buffer,
            layout,
            bind_group,
        }
    }

    #[inline(always)]
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    #[inline(always)]
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn update(&self, queue: &wgpu::Queue, camera: &Camera) {
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[CameraUniform {
                view_proj: camera.compute_view_proj(),
            }]),
        );
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: glm::Mat4,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: glm::identity::<f32, 4>(),
        }
    }
}
