use glam::Mat4;
use wgpu::util::DeviceExt;

use super::BasePipelineBuffer;
use crate::renderer::managers::mesh_manager::InternalMesh;
use crate::renderer::types::{Texture, Vertex};

pub struct GeometryPipeline {
    pipeline: wgpu::RenderPipeline,
    instance_bind_group_layout: wgpu::BindGroupLayout,
}

impl GeometryPipeline {
    pub fn new(device: &wgpu::Device, base_pipeline_buffer: &BasePipelineBuffer) -> Self {
        let instance_bind_group_layout = InstanceUniform::make_layout(device);

        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[base_pipeline_buffer.layout(), &instance_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self {
            pipeline,
            instance_bind_group_layout,
        }
    }

    pub fn create_instance_description(
        &self,
        device: &wgpu::Device,
        transform: &Mat4,
        texture: &Texture,
    ) -> InstanceDescription {
        InstanceDescription::new(device, &self.instance_bind_group_layout, transform, texture)
    }

    pub fn render<'a, I>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, meshes: I)
    where
        I: Iterator<Item = (&'a InternalMesh, &'a InstanceDescription)>,
    {
        render_pass.set_pipeline(&self.pipeline);
        meshes.for_each(|(mesh, descr)| {
            render_pass.set_bind_group(1, &descr.bind_group, &[]);
            render_pass.draw_indexed(mesh.indices(), mesh.vertex_range.start as i32, 0..1);
        })
    }
}

pub struct InstanceDescription {
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

impl InstanceDescription {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        transform: &Mat4,
        texture: &Texture,
    ) -> Self {
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[InstanceUniform::new(transform)]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        uniform_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });

        Self {
            bind_group,
            uniform_buffer,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, transform: &Mat4) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[InstanceUniform::new(transform)]),
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceUniform {
    transform: [f32; 4 * 3],
    normal_matrix: [f32; 4 * 3],
}

impl InstanceUniform {
    fn new(transform: &Mat4) -> Self {
        let normal_matrix = transform.inverse().transpose();

        InstanceUniform {
            transform: transform.as_ref()[..4 * 3].try_into().unwrap(),
            normal_matrix: normal_matrix.as_ref()[..4 * 3].try_into().unwrap(),
        }
    }

    fn make_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
}
