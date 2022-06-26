use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl Mesh {
    pub fn quad(device: &wgpu::Device) -> Self {
        pub const TRIANGLE: &[Vertex] = &[
            Vertex {
                position: [-0.5, 0.5, 0.0],
                tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: [-0.5, -0.5, 0.0],
                tex_coords: [0.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0],
                tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.0],
                tex_coords: [1.0, 0.0],
            },
        ];

        pub const INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

        Self::new(device, TRIANGLE, INDICES)
    }

    pub fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u16]) -> Self {
        Self {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quad_vertex_buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quad_index_buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: indices.len() as u32,
        }
    }
}
