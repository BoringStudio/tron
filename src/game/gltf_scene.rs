use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;

use anyhow::{Context, Result};
use glam::{Vec2, Vec3};

use crate::renderer::managers::MeshManager;
use crate::renderer::types::{Mesh, MeshHandle, Vertex};
use crate::renderer::CommandEncoder;

pub fn load(
    ctx: &mut CommandEncoder<'_>,
    mesh_manager: &mut MeshManager,
    resource_id: &AtomicUsize,
    data: &[u8],
) -> Result<HashMap<String, MeshHandle>> {
    let (file, buffers, _) = gltf::import_slice(data)?;
    let scene = file.default_scene().context("Default scene not found")?;

    let mut meshes = HashMap::new();

    let mut stack = Vec::from([Segment {
        node: scene.nodes().next().context("Root node not found")?,
        path: String::new(),
    }]);

    while let Some(segment) = stack.pop() {
        if let Some(mesh) = segment.node.mesh() {
            let mesh_name = mesh.name();

            for (i, primitive) in mesh.primitives().enumerate() {
                let mesh = load_mesh(ctx, mesh_manager, resource_id, &buffers, primitive)?;
                meshes.insert(segment.primitive_path(mesh_name, i), mesh);
            }
        }

        for (i, child) in segment.node.children().enumerate() {
            let path = segment.child_path(child.name(), i);
            stack.push(Segment { node: child, path });
        }
    }

    Ok(meshes)
}

struct Segment<'a> {
    node: gltf::Node<'a>,
    path: String,
}

impl Segment<'_> {
    fn child_path(&self, name: Option<&str>, i: usize) -> String {
        match name {
            Some(name) => format!("{}/{name}", self.path),
            None => format!("{}/{i}", self.path),
        }
    }

    fn primitive_path(&self, mesh_name: Option<&str>, i: usize) -> String {
        match mesh_name {
            Some(name) => format!("{}/{name}/{i}", self.path),
            None => format!("{}/{i}", self.path),
        }
    }
}

fn load_mesh(
    ctx: &mut CommandEncoder<'_>,
    mesh_manager: &mut MeshManager,
    resource_id: &AtomicUsize,
    buffers: &[gltf::buffer::Data],
    primitive: gltf::Primitive<'_>,
) -> Result<MeshHandle> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(std::ops::Deref::deref));
    let positions = reader.read_positions().context("Positions not found")?;
    let vertex_count = positions.len();

    fn optional_iter<I, T: Default>(
        iter: Option<I>,
        len: usize,
    ) -> Result<(bool, impl Iterator<Item = T>)>
    where
        I: Iterator<Item = T> + ExactSizeIterator,
    {
        match iter {
            Some(iter) => {
                anyhow::ensure!(iter.len() == len, "component array length mismatch");
                Ok((true, either::Left(iter)))
            }
            None => Ok((
                false,
                either::Right(std::iter::repeat_with(T::default).take(len)),
            )),
        }
    }

    let (has_normals, normals) = optional_iter(reader.read_normals(), vertex_count)?;
    let (has_tangents, tangents) = optional_iter(reader.read_tangents(), vertex_count)?;
    let (has_uvs, uvs0) = optional_iter(
        reader.read_tex_coords(0).map(|iter| iter.into_f32()),
        vertex_count,
    )?;

    let mut vertices = Vec::with_capacity(positions.len());
    for (((position, normal), tangents), uv0) in positions.zip(normals).zip(tangents).zip(uvs0) {
        vertices.push(Vertex {
            position: Vec3::from_array(position),
            normal: Vec3::from_array(normal),
            tangent: Vec3::new(tangents[0], tangents[1], tangents[2]),
            uv0: Vec2::from_array(uv0),
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

    let mut mesh = Mesh { vertices, indices };
    mesh.validate()?;

    if !has_normals {
        // SAFETY: mesh has already been validated
        unsafe { mesh.compute_normals() };
    }

    if !has_tangents && has_uvs {
        // SAFETY: mesh has already been validated
        unsafe { mesh.compute_tangents() };
    }

    let handle = MeshManager::allocate(&resource_id);
    mesh_manager.set_mesh(ctx, &handle, mesh);

    Ok(handle)
}
