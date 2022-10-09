use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::mesh::{Mesh, Vertex};

pub struct SceneUniformBuffer {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl SceneUniformBuffer {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene_uniform_buffer"),
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

pub fn load_object(device: &wgpu::Device, data: &[u8]) -> Result<Vec<Mesh>> {
    let (file, buffers, _) = gltf::import_slice(data)?;
    let scene = file.default_scene().context("Default scene not found")?;

    let mut meshes = Vec::new();

    let mut stack = Vec::from([scene.nodes().next().context("Root node not found")?]);
    while let Some(node) = stack.pop() {
        println!("NODE: {}, {:?}", node.index(), node.name());

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                meshes.push(load_mesh(device, &buffers, primitive)?);
            }
        }

        for child in node.children() {
            stack.push(child);
        }
    }

    Ok(meshes)
}

fn load_mesh(
    device: &wgpu::Device,
    buffers: &[gltf::buffer::Data],
    primitive: gltf::Primitive<'_>,
) -> Result<Mesh> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(std::ops::Deref::deref));
    let positions = reader.read_positions().context("Positions not found")?;

    let normals = reader.read_normals().context("Normals not found")?;
    anyhow::ensure!(normals.len() == positions.len(), "Normal count mismatch");

    let uvs0 = reader
        .read_tex_coords(0)
        .context("Text coords not found")?
        .into_f32();
    anyhow::ensure!(uvs0.len() == positions.len(), "UV0 count mismatch");

    let mut vertices = Vec::with_capacity(positions.len());
    for ((position, normal), uv0) in positions.zip(normals).zip(uvs0) {
        vertices.push(Vertex {
            position,
            normal,
            uv0,
        });
    }

    let mesh_indices = reader
        .read_indices()
        .context("Indices not found")?
        .into_u32();

    let mut indices = Vec::with_capacity(mesh_indices.len());
    for index in mesh_indices {
        indices.push(index);
    }

    Ok(Mesh::new(device, &vertices, &indices))
}
