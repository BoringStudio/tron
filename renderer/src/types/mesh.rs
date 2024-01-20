use anyhow::Result;
use glam::{Vec2, Vec3};

use crate::resource_handle::{RawResourceHandle, ResourceHandle};
use crate::types::{BoundingSphere, Color, Normal, Position, Tangent, VertexAttributeData, UV0};

pub type MeshHandle = ResourceHandle<Mesh>;
pub type RawMeshHandle = RawResourceHandle<Mesh>;

pub struct Mesh {
    vertex_count: u32,
    attribute_data: Vec<VertexAttributeData>,
    indices: Vec<u32>,
    bounding_sphere: BoundingSphere,
}

impl Mesh {
    pub fn builder<T: MeshGenerator>(generator: T) -> MeshBuilder {
        generator.generate_mesh()
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    pub fn attribute_data(&self) -> &[VertexAttributeData] {
        &self.attribute_data
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn bounding_sphere(&self) -> &BoundingSphere {
        &self.bounding_sphere
    }
}

pub trait MeshGenerator: Sized {
    fn generate_mesh(self) -> MeshBuilder;
}

impl MeshGenerator for Vec<Position> {
    fn generate_mesh(self) -> MeshBuilder {
        MeshBuilder::new(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlaneMeshGenerator {
    pub width: f32,
    pub height: f32,
}

impl PlaneMeshGenerator {
    pub fn from_size(side_size: f32) -> Self {
        Self::from_extent(Vec2::splat(side_size))
    }

    pub fn from_extent(extent: Vec2) -> Self {
        Self {
            width: extent.x,
            height: extent.y,
        }
    }
}

impl Default for PlaneMeshGenerator {
    #[inline]
    fn default() -> Self {
        Self::from_size(1.0)
    }
}

impl MeshGenerator for PlaneMeshGenerator {
    fn generate_mesh(self) -> MeshBuilder {
        //  ^y
        //  |/z
        //  -> x
        //     width
        //   2-----3
        //  /  *  / height
        // 0-----1

        let half_x = self.width * 0.5;
        let half_z = self.height * 0.5;

        let positions = vec![
            Position(Vec3::new(-half_x, 0.0, -half_z)),
            Position(Vec3::new(half_x, 0.0, -half_z)),
            Position(Vec3::new(-half_x, 0.0, half_z)),
            Position(Vec3::new(half_x, 0.0, half_z)),
        ];
        let uv0 = vec![
            UV0(Vec2::new(0.0, 0.0)),
            UV0(Vec2::new(1.0, 0.0)),
            UV0(Vec2::new(0.0, 1.0)),
            UV0(Vec2::new(1.0, 1.0)),
        ];
        let indices = vec![0, 2, 3, 0, 3, 1];

        MeshBuilder::new(positions)
            .with_uv0(uv0)
            .with_indices(indices)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CubeMeshGenerator {
    pub a: Vec3,
    pub b: Vec3,
}

impl CubeMeshGenerator {
    pub fn from_size(side_size: f32) -> Self {
        Self::from_extent(Vec3::splat(side_size))
    }

    pub fn from_extent(extent: Vec3) -> Self {
        Self {
            a: Vec3::new(-extent.x / 2.0, -extent.y / 2.0, -extent.z / 2.0),
            b: Vec3::new(extent.x / 2.0, extent.y / 2.0, extent.z / 2.0),
        }
    }

    pub fn from_points(a: Vec3, b: Vec3) -> Self {
        Self { a, b }
    }
}

impl Default for CubeMeshGenerator {
    #[inline]
    fn default() -> Self {
        Self::from_size(1.0)
    }
}

impl MeshGenerator for CubeMeshGenerator {
    fn generate_mesh(self) -> MeshBuilder {
        //  ^y
        //  |/z
        //  -> x
        //
        //     6----7 b
        //    /|   /|
        //   2----3 |
        //   | 4--|-5
        //   |/   |/
        // a 0----1
        //
        let positions = vec![
            // Front
            Position(Vec3::new(self.a.x, self.a.y, self.b.z)), // 0
            Position(Vec3::new(self.b.x, self.a.y, self.a.z)), // 1
            Position(Vec3::new(self.a.x, self.b.y, self.a.z)), // 2
            Position(Vec3::new(self.b.x, self.b.y, self.a.z)), // 3
            // Right
            Position(Vec3::new(self.b.x, self.a.y, self.a.z)), // 1
            Position(Vec3::new(self.b.x, self.a.y, self.b.z)), // 5
            Position(Vec3::new(self.b.x, self.b.y, self.a.z)), // 3
            Position(Vec3::new(self.b.x, self.b.y, self.b.z)), // 7
            // Back
            Position(Vec3::new(self.b.x, self.a.y, self.b.z)), // 5
            Position(Vec3::new(self.a.x, self.a.y, self.b.z)), // 4
            Position(Vec3::new(self.b.x, self.b.y, self.b.z)), // 7
            Position(Vec3::new(self.a.x, self.b.y, self.b.z)), // 6
            // Left
            Position(Vec3::new(self.a.x, self.a.y, self.b.z)), // 4
            Position(Vec3::new(self.a.x, self.a.y, self.a.z)), // 0
            Position(Vec3::new(self.a.x, self.b.y, self.b.z)), // 6
            Position(Vec3::new(self.a.x, self.b.y, self.a.z)), // 2
            // Top
            Position(Vec3::new(self.a.x, self.b.y, self.a.z)), // 2
            Position(Vec3::new(self.b.x, self.b.y, self.a.z)), // 3
            Position(Vec3::new(self.a.x, self.b.y, self.b.z)), // 6
            Position(Vec3::new(self.b.x, self.b.y, self.b.z)), // 7
            // Bottom
            Position(Vec3::new(self.a.x, self.a.y, self.a.z)), // 0
            Position(Vec3::new(self.b.x, self.a.y, self.a.z)), // 1
            Position(Vec3::new(self.a.x, self.a.y, self.b.z)), // 4
            Position(Vec3::new(self.b.x, self.a.y, self.b.z)), // 5
        ];

        // List of cube UVs:
        let uv0 = vec![
            // Front
            UV0(Vec2::new(0.0, 0.0)), // 0
            UV0(Vec2::new(1.0, 0.0)), // 1
            UV0(Vec2::new(0.0, 1.0)), // 2
            UV0(Vec2::new(1.0, 1.0)), // 3
            // Right
            UV0(Vec2::new(0.0, 0.0)), // 1
            UV0(Vec2::new(1.0, 0.0)), // 5
            UV0(Vec2::new(0.0, 1.0)), // 3
            UV0(Vec2::new(1.0, 1.0)), // 7
            // Back
            UV0(Vec2::new(0.0, 0.0)), // 5
            UV0(Vec2::new(1.0, 0.0)), // 4
            UV0(Vec2::new(0.0, 1.0)), // 7
            UV0(Vec2::new(1.0, 1.0)), // 6
            // Left
            UV0(Vec2::new(0.0, 0.0)), // 4
            UV0(Vec2::new(1.0, 0.0)), // 0
            UV0(Vec2::new(0.0, 1.0)), // 6
            UV0(Vec2::new(1.0, 1.0)), // 2
            // Top
            UV0(Vec2::new(0.0, 0.0)), // 2
            UV0(Vec2::new(1.0, 0.0)), // 3
            UV0(Vec2::new(0.0, 1.0)), // 6
            UV0(Vec2::new(1.0, 1.0)), // 7
            // Bottom
            UV0(Vec2::new(0.0, 0.0)), // 0
            UV0(Vec2::new(1.0, 0.0)), // 1
            UV0(Vec2::new(0.0, 1.0)), // 4
            UV0(Vec2::new(1.0, 1.0)), // 5
        ];

        // List of cube indices:
        let indices = vec![
            0, 2, 3, 0, 3, 1, // front
            4, 6, 7, 4, 7, 5, // right
            8, 10, 11, 8, 11, 9, // back
            12, 14, 15, 12, 15, 13, // left
            16, 18, 19, 16, 19, 17, // top
            20, 22, 23, 20, 23, 21, // bottom
        ];

        MeshBuilder::new(positions)
            .with_uv0(uv0)
            .with_indices(indices)
    }
}

#[derive(Default)]
pub struct MeshBuilder {
    vertex_count: usize,
    positions: Vec<Position>,
    normals: Option<ComputableData<Vec<Normal>>>,
    tangents: Option<ComputableData<Vec<Tangent>>>,
    uv0: Option<Vec<UV0>>,
    colors: Option<Vec<Color>>,

    indices: Option<Vec<u32>>,
    double_sided: bool,
}

impl MeshBuilder {
    pub fn new(positions: Vec<Position>) -> Self {
        Self {
            vertex_count: positions.len(),
            positions,
            ..Self::default()
        }
    }

    pub fn with_normals(mut self, normals: Vec<Normal>) -> Self {
        self.normals = Some(ComputableData::Known(normals));
        self
    }

    pub fn with_computed_normals(mut self) -> Self {
        self.normals = Some(ComputableData::Compute);
        self
    }

    pub fn with_tangents(mut self, tangents: Vec<Tangent>) -> Self {
        self.tangents = Some(ComputableData::Known(tangents));
        self
    }

    pub fn with_computed_tangents(mut self) -> Self {
        self.tangents = Some(ComputableData::Compute);
        self
    }

    pub fn with_uv0(mut self, uv0: Vec<UV0>) -> Self {
        self.uv0 = Some(uv0);
        self
    }

    pub fn with_colors(mut self, colors: Vec<Color>) -> Self {
        self.colors = Some(colors);
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

    pub fn build(self) -> Result<Mesh> {
        let len = self.vertex_count;

        if matches!(&self.normals, Some(ComputableData::Known(v)) if v.len() != len)
            || matches!(&self.tangents, Some(ComputableData::Known(v)) if v.len() != len)
            || matches!(&self.uv0, Some(v) if v.len() != len)
            || matches!(&self.colors, Some(v) if v.len() != len)
        {
            anyhow::bail!("component length mismatch");
        }

        let mut indices = self.indices.unwrap_or_else(|| (0..len as u32).collect());

        anyhow::ensure!(len <= indices.len(), "index count mismatch");
        anyhow::ensure!(
            indices.len() % 3 == 0,
            "index count must be a multiple of 3"
        );

        anyhow::ensure!(
            indices.iter().all(|index| (*index as usize) < len),
            "indices must not exceed vertex count"
        );

        if matches!(
            &self.tangents,
            Some(ComputableData::Compute) if self.normals.is_none() || self.uv0.is_none()
        ) {
            anyhow::bail!("tangents can only be computed if normals and uv0 is present");
        }

        if self.double_sided {
            // SAFETY: `indices` were checked to be valid above.
            unsafe { make_double_sided(&mut indices) };
        }

        let normals = match self.normals {
            Some(ComputableData::Known(normals)) => Some(normals),
            Some(ComputableData::Compute) => {
                // SAFETY: `indices` were checked to be valid above.
                Some(unsafe { compute_normals(&indices, &self.positions) })
            }
            None => None,
        };

        let tangents = match (self.tangents, &normals, &self.uv0) {
            (None, _, _) => None,
            (Some(ComputableData::Known(tangents)), _, _) => Some(tangents),
            (Some(ComputableData::Compute), Some(normals), Some(uv)) => {
                // SAFETY: `indices`, `normals` and `uv` were checked to be valid above.
                Some(unsafe { compute_tangents(&indices, &self.positions, normals, uv) })
            }
            _ => unreachable!(),
        };

        let bounding_sphere = BoundingSphere::compute_from_positions(&self.positions);

        let mut attribute_data = Vec::with_capacity(
            1 + normals.is_some() as usize
                + tangents.is_some() as usize
                + self.uv0.is_some() as usize
                + self.colors.is_some() as usize,
        );

        attribute_data.push(VertexAttributeData::new(self.positions));
        if let Some(normals) = normals {
            attribute_data.push(VertexAttributeData::new(normals));
        }
        if let Some(tangents) = tangents {
            attribute_data.push(VertexAttributeData::new(tangents));
        }
        if let Some(uv0) = self.uv0 {
            attribute_data.push(VertexAttributeData::new(uv0));
        }
        if let Some(colors) = self.colors {
            attribute_data.push(VertexAttributeData::new(colors));
        }

        Ok(Mesh {
            vertex_count: len as u32,
            attribute_data,
            indices,
            bounding_sphere,
        })
    }
}

enum ComputableData<T> {
    Known(T),
    Compute,
}

/// # Safety
/// The following must be true:
/// - `indices` must have a length equal to a multiple of 3.
/// - `indices` must be in a valid range for `positions`.
unsafe fn make_double_sided(indices: &mut Vec<u32>) {
    let len = indices.len();
    let triangle_count = len / 3;

    indices.reserve(len);

    let ptr = indices.as_mut_ptr();

    unsafe {
        // Iterate in reverse as to not overwrite indices.
        for i in (0..triangle_count).rev() {
            let i1 = *ptr.add(i * 3);
            let i2 = *ptr.add(i * 3 + 1);
            let i3 = *ptr.add(i * 3 + 2);

            // One triangle forward.
            ptr.add(i * 6).write(i1);
            ptr.add(i * 6 + 1).write(i2);
            ptr.add(i * 6 + 2).write(i3);

            // One triangle reverse.
            ptr.add(i * 6 + 3).write(i3);
            ptr.add(i * 6 + 4).write(i2);
            ptr.add(i * 6 + 5).write(i1);
        }

        indices.set_len(len * 2);
    }
}

/// # Safety
/// The following must be true:
/// - `indices` must have a length equal to a multiple of 3.
/// - `indices` must be in a valid range for `positions`.
unsafe fn compute_normals(indices: &[u32], positions: &[Position]) -> Vec<Normal> {
    let mut normals = vec![Normal::ZERO; positions.len()];

    for idx in indices.chunks_exact(3) {
        let (idx0, idx1, idx2) = match *idx {
            [idx0, idx1, idx2] => (idx0, idx1, idx2),
            _ => std::hint::unreachable_unchecked(),
        };

        let pos0 = positions.get_unchecked(idx0 as usize).0;
        let pos1 = positions.get_unchecked(idx1 as usize).0;
        let pos2 = positions.get_unchecked(idx2 as usize).0;

        let edge0 = pos1 - pos0;
        let edge1 = pos2 - pos0;

        let normal = edge0.cross(edge1);

        normals.get_unchecked_mut(idx0 as usize).0 += normal;
        normals.get_unchecked_mut(idx1 as usize).0 += normal;
        normals.get_unchecked_mut(idx2 as usize).0 += normal;
    }

    for normal in &mut normals {
        normal.0 = normal.0.normalize_or_zero();
    }

    normals
}

/// # Safety
/// The following must be true:
/// - `indices` must have a length equal to a multiple of 3.
/// - `indices` must be in a valid range for `positions`.
/// - `normals` must have a length equal to `positions`.
/// - `uv` must have a length equal to `positions`.
unsafe fn compute_tangents(
    indices: &[u32],
    positions: &[Position],
    normals: &[Normal],
    uv: &[UV0],
) -> Vec<Tangent> {
    let mut tangents = vec![Tangent::ZERO; positions.len()];

    for idx in indices.chunks_exact(3) {
        let (idx0, idx1, idx2) = match *idx {
            [idx0, idx1, idx2] => (idx0, idx1, idx2),
            _ => std::hint::unreachable_unchecked(),
        };

        let pos0 = positions.get_unchecked(idx0 as usize).0;
        let pos1 = positions.get_unchecked(idx1 as usize).0;
        let pos2 = positions.get_unchecked(idx2 as usize).0;

        let uv0 = uv.get_unchecked(idx0 as usize).0;
        let uv1 = uv.get_unchecked(idx1 as usize).0;
        let uv2 = uv.get_unchecked(idx2 as usize).0;

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

        tangents.get_unchecked_mut(idx0 as usize).0 += tangent;
        tangents.get_unchecked_mut(idx1 as usize).0 += tangent;
        tangents.get_unchecked_mut(idx2 as usize).0 += tangent;
    }

    for (tangent, normal) in tangents.iter_mut().zip(normals) {
        tangent.0 = (tangent.0 - (normal.0 * normal.0.dot(tangent.0))).normalize_or_zero();
    }

    tangents
}
