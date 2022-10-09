use wgpu::util::DeviceExt;

use crate::scene::SceneUniformBuffer;

pub struct SkyPipeline {
    pipeline: wgpu::RenderPipeline,
    sky_uniform_buffer: SkyUniformBuffer,
    sky_box: SkyBox,
}

impl SkyPipeline {
    pub fn new(device: &wgpu::Device, scene_uniform_buffer: &SceneUniformBuffer) -> Self {
        let sky_uniform_buffer = SkyUniformBuffer::new(device);

        let sky_box = SkyBox::new(device);

        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sky.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_pipeline_layout"),
            bind_group_layouts: &[scene_uniform_buffer.layout(), sky_uniform_buffer.layout()],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: "vs_main",
                buffers: &[SkyBoxVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
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
            sky_uniform_buffer,
            sky_box,
        }
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);

        render_pass.set_bind_group(1, self.sky_uniform_buffer.bind_group(), &[]);

        render_pass.set_vertex_buffer(0, self.sky_box.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.sky_box.index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        render_pass.draw_indexed(0..self.sky_box.index_count, 0, 0..1);
    }
}

struct SkyUniformBuffer {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl SkyUniformBuffer {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky_uniform_buffer"),
            contents: bytemuck::cast_slice(&[SkyUniform::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_bind_group"),
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
    fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    #[inline(always)]
    fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniform {
    rayleigh: f32,
    turbidity: f32,
    mie_coefficient: f32,
    luminance: f32,
    direction: glm::Vec3,
    mie_directional_g: f32,
}

impl Default for SkyUniform {
    fn default() -> Self {
        Self {
            turbidity: 10.0,
            rayleigh: 3.0,
            mie_coefficient: 0.005,
            luminance: 0.5,
            direction: glm::normalize(&glm::Vec3::new(1.0, -1.0, 0.0)),
            mie_directional_g: 0.7,
        }
    }
}

// NOTE: `assert_eq` doesn't work in const context
const _: () = assert!(std::mem::size_of::<SkyUniform>() == 32);

struct SkyBox {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl SkyBox {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sky_box_vertex_buffer"),
                contents: bytemuck::cast_slice(SKY_BOX_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sky_box_index_buffer"),
                contents: bytemuck::cast_slice(SKY_BOX_INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: SKY_BOX_INDICES.len() as u32,
        }
    }
}

const SKY_BOX_VERTICES: &[SkyBoxVertex] = &[
    SkyBoxVertex::new(-1.0, 1.0, 1.0),   // 0
    SkyBoxVertex::new(1.0, 1.0, 1.0),    // 1
    SkyBoxVertex::new(1.0, -1.0, 1.0),   // 2
    SkyBoxVertex::new(-1.0, -1.0, 1.0),  // 3
    SkyBoxVertex::new(-1.0, 1.0, -1.0),  // 4
    SkyBoxVertex::new(1.0, 1.0, -1.0),   // 5
    SkyBoxVertex::new(1.0, -1.0, -1.0),  // 6
    SkyBoxVertex::new(-1.0, -1.0, -1.0), // 7
];

const SKY_BOX_INDICES: &[u16] = &[
    0, 2, 1, 0, 3, 2, //
    1, 6, 5, 1, 2, 6, //
    5, 7, 4, 5, 6, 7, //
    4, 3, 0, 4, 7, 3, //
    1, 5, 4, 1, 4, 0, //
    2, 7, 6, 2, 3, 7, //
];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyBoxVertex {
    pos: glm::Vec3,
}

impl SkyBoxVertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
        0 => Float32x3,
    ];

    #[inline(always)]
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            pos: glm::Vec3::new(x, y, z),
        }
    }

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Self::ATTRIBUTES,
        }
    }
}
