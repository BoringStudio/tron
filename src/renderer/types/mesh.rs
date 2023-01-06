use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

#[derive(Copy, Clone, Debug)]
pub enum VertexBufferType {
    Position,
    Normal,
    Tangent,
    Uv0,
}

#[derive(Default)]
pub struct MeshBuilder {
    vertex_count: usize,
    positions: Vec<Vec3>,
    normals: Option<Vec<Vec3>>,
    tangents: Option<Vec<Vec3>>,
    uv0: Option<Vec<Vec2>>,

    indices: Option<Vec<u32>>,
    double_sided: bool,
}

impl MeshBuilder {
    pub fn new(positions: Vec<Vec3>) -> Self {
        Self {
            vertex_count: positions.len(),
            positions,
            ..Self::default()
        }
    }

    pub fn with_normals(mut self, normals: Vec<Vec3>) -> Self {
        self.normals = Some(normals);
        self
    }

    pub fn with_tangents(mut self, tangents: Vec<Vec3>) -> Self {
        self.tangents = Some(tangents);
        self
    }

    pub fn with_uv0(mut self, uv0: Vec<Vec2>) -> Self {
        self.uv0 = Some(uv0);
        self
    }

    pub fn with_indices(mut self, indices: Vec<u32>) -> Self {
        self.indices = Some(indices);
        self
    }

    pub fn double_sided(mut self) -> Self {
        self.double_sided = true;
        self
    }

    pub fn build(self) -> Result<Mesh, MeshValidationError> {
        fn make_component_iter<T: Default>(
            items: Option<Vec<T>>,
            len: usize,
        ) -> impl Iterator<Item = T> + ExactSizeIterator {
            match items {
                Some(items) => either::Left(items.into_iter()),
                None => either::Right((0..len).map(|_| T::default())),
            }
        }

        let len = self.vertex_count;

        let has_normals = self.normals.is_some();
        let has_tangents = self.tangents.is_some();
        let has_uvs = self.uv0.is_some();

        if matches!(&self.normals, Some(v) if v.len() != len)
            || matches!(&self.tangents, Some(v) if v.len() != len)
            || matches!(&self.uv0, Some(v) if v.len() != len)
        {
            return Err(MeshValidationError::ComponentMismatch);
        }

        let mut vertices = Vec::with_capacity(len);
        for (((position, normal), tangent), uv0) in self
            .positions
            .into_iter()
            .zip(make_component_iter(self.normals, len))
            .zip(make_component_iter(self.tangents, len))
            .zip(make_component_iter(self.uv0, len))
        {
            vertices.push(Vertex {
                position,
                normal,
                tangent,
                uv0,
            });
        }

        let mut mesh = Mesh {
            vertices,
            indices: self.indices.unwrap_or_else(|| (0..len as u32).collect()),
        };

        mesh.validate()?;

        if !has_normals {
            // SAFETY: mesh was validated before
            unsafe { mesh.compute_normals() };
        }

        if !has_tangents && has_uvs {
            // SAFETY: mesh was validated before
            unsafe { mesh.compute_tangents() };
        }

        Ok(mesh)
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn validate(&self) -> Result<(), MeshValidationError> {
        let vertex_count = self.vertices.len();
        let index_count = self.indices.len();

        if vertex_count > index_count {
            return Err(MeshValidationError::IndexCountMismatch);
        }

        if index_count % 3 != 0 {
            return Err(MeshValidationError::InvalidIndexCount);
        }

        if self
            .indices
            .iter()
            .any(|index| *index as usize >= vertex_count)
        {
            return Err(MeshValidationError::IndexOutOfBounds);
        }

        Ok(())
    }

    pub unsafe fn compute_normals(&mut self) {
        debug_assert_eq!(self.indices.len() % 3, 0);

        for idx in self.indices.chunks_exact(3) {
            let (idx0, idx1, idx2) = match *idx {
                [idx0, idx1, idx2] => (idx0, idx1, idx2),
                _ => std::hint::unreachable_unchecked(),
            };

            let pos0 = self.vertices.get_unchecked(idx0 as usize).position;
            let pos1 = self.vertices.get_unchecked(idx1 as usize).position;
            let pos2 = self.vertices.get_unchecked(idx2 as usize).position;

            let edge0 = pos1 - pos0;
            let edge1 = pos2 - pos0;

            let normal = edge0.cross(edge1);

            self.vertices.get_unchecked_mut(idx0 as usize).normal += normal;
            self.vertices.get_unchecked_mut(idx1 as usize).normal += normal;
            self.vertices.get_unchecked_mut(idx2 as usize).normal += normal;
        }

        for vertex in &mut self.vertices {
            vertex.normal = vertex.normal.normalize_or_zero();
        }
    }

    pub unsafe fn compute_tangents(&mut self) {
        debug_assert_eq!(self.indices.len() % 3, 0);

        for idx in self.indices.chunks_exact(3) {
            let (idx0, idx1, idx2) = match *idx {
                [idx0, idx1, idx2] => (idx0, idx1, idx2),
                _ => std::hint::unreachable_unchecked(),
            };

            let v0 = *self.vertices.get_unchecked(idx0 as usize);
            let v1 = *self.vertices.get_unchecked(idx1 as usize);
            let v2 = *self.vertices.get_unchecked(idx2 as usize);

            let pos_edge0 = v1.position - v0.position;
            let pos_edge1 = v2.position - v0.position;

            let uv_edge0 = v1.uv0 - v0.uv0;
            let uv_edge1 = v2.uv0 - v0.uv0;

            let r = 1.0 / (uv_edge0.x * uv_edge1.y - uv_edge0.y * uv_edge1.x);

            let tangent = Vec3::new(
                (pos_edge0.x * uv_edge1.y - pos_edge1.x * uv_edge0.y) * r,
                (pos_edge0.y * uv_edge1.y - pos_edge1.y * uv_edge0.y) * r,
                (pos_edge0.z * uv_edge1.y - pos_edge1.z * uv_edge0.y) * r,
            );

            self.vertices.get_unchecked_mut(idx0 as usize).tangent += tangent;
            self.vertices.get_unchecked_mut(idx1 as usize).tangent += tangent;
            self.vertices.get_unchecked_mut(idx2 as usize).tangent += tangent;
        }

        for vertex in &mut self.vertices {
            vertex.tangent = (vertex.tangent - (vertex.normal * vertex.normal.dot(vertex.tangent)))
                .normalize_or_zero();
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MeshValidationError {
    #[error("component array length mismatch")]
    ComponentMismatch,
    #[error("more vertices than indices")]
    IndexCountMismatch,
    #[error("index count is not multiple of three")]
    InvalidIndexCount,
    #[error("index out of bounds")]
    IndexOutOfBounds,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub uv0: Vec2,
}

impl Vertex {
    const ATTRIBUTES: &'static [wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x3,
        3 => Float32x2,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Self::ATTRIBUTES,
        }
    }
}
