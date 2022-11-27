use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use wgpu::util::DeviceExt;

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
        let len = self.vertex_count;

        let has_normals = self.normals.is_some();
        let has_tangents = self.tangents.is_some();
        let has_uvs = self.uv0.is_some();

        if let Some(normals) = self.normals {
            normals.
        }

        let mut mesh = Mesh {
            positions: self.positions,
            normals: self.normals.unwrap_or_else(|| vec![Vec3::ZERO; len]),
            tangents: self.tangents.unwrap_or_else(|| vec![Vec3::ZERO; len]),
            uv0: self.uv0.unwrap_or_else(|| vec![Vec2::ZERO; len]),
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
        let vertex_count = self.positions.len();
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

    unsafe fn compute_normals(&mut self) {
        debug_assert_eq!(self.indices.len() % 3, 0);
        debug_assert_eq!(self.normals.len(), self.positions.len());

        for idx in self.indices.chunks_exact(3) {
            let (idx0, idx1, idx2) = match *idx {
                [idx0, idx1, idx2] => (idx0, idx1, idx2),
                _ => std::hint::unreachable_unchecked(),
            };

            let pos0 = *self.positions.get_unchecked(idx0 as usize);
            let pos1 = *self.positions.get_unchecked(idx1 as usize);
            let pos2 = *self.positions.get_unchecked(idx2 as usize);

            let edge0 = pos1 - pos0;
            let edge1 = pos2 - pos0;

            let normal = edge0.cross(edge1);

            *self.normals.get_unchecked_mut(idx0 as usize) += normal;
            *self.normals.get_unchecked_mut(idx1 as usize) += normal;
            *self.normals.get_unchecked_mut(idx2 as usize) += normal;
        }

        for normal in &mut self.normals {
            *normal = normal.normalize_or_zero();
        }
    }

    unsafe fn compute_tangents(&mut self) {
        debug_assert_eq!(self.indices.len() % 3, 0);
        debug_assert_eq!(self.tangents.len(), self.positions.len());

        for idx in self.indices.chunks_exact(3) {
            let (idx0, idx1, idx2) = match *idx {
                [idx0, idx1, idx2] => (idx0, idx1, idx2),
                _ => std::hint::unreachable_unchecked(),
            };

            let pos0 = *self.positions.get_unchecked(idx0 as usize);
            let pos1 = *self.positions.get_unchecked(idx1 as usize);
            let pos2 = *self.positions.get_unchecked(idx2 as usize);

            let uv0 = *self.uv0.get_unchecked(idx0 as usize);
            let uv1 = *self.uv0.get_unchecked(idx1 as usize);
            let uv2 = *self.uv0.get_unchecked(idx2 as usize);

            let pos_edge0 = pos1 - pos0;
            let pos_edge1 = pos2 - pos0;

            let uv_edge0 = uv1 - uv0;
            let uv_edge1 = uv2 - uv0;

            let r = 1.0 / (uv_edge0.x * uv_edge1.y - uv_edge0.y * uv_edge1.x);

            let tangent = Vec3::new(
                (pos_edge0.x * uv_edge1.y - pos_edge1.x * uv_edge0.y) * r,
                (pos_edge0.y * uv_edge1.y - pos_edge1.y * uv_edge0.y) * r,
                (pos_edge0.z * uv_edge1.y - pos_edge1.z * uv_edge0.y) * r,
            );

            *self.tangents.get_unchecked_mut(idx0 as usize) += tangent;
            *self.tangents.get_unchecked_mut(idx1 as usize) += tangent;
            *self.tangents.get_unchecked_mut(idx2 as usize) += tangent;
        }

        for (tangent, normal) in self.tangents.iter_mut().zip(self.normals.iter()) {
            *tangent = (*tangent - (*normal * normal.dot(*tangent))).normalize_or_zero();
        }
    }
}

#[derive(thiserror::Error, Debug)]
enum MeshValidationError {
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
