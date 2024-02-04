use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

use crate::types::Position;

/// Frustum of a camera with infinite far plane.
#[derive(Debug, Clone, gfx::AsStd140, gfx::AsStd430)]
pub struct Frustum {
    pub near: Plane,
    pub left: Plane,
    pub right: Plane,
    pub top: Plane,
    pub bottom: Plane,
}

impl Frustum {
    pub const IDENTITY: Self = Self {
        near: Plane {
            normal: Vec3::Z,
            distance: 1.0,
        },
        left: Plane {
            normal: Vec3::NEG_X,
            distance: 1.0,
        },
        right: Plane {
            normal: Vec3::X,
            distance: 1.0,
        },
        top: Plane {
            normal: Vec3::Y,
            distance: 1.0,
        },
        bottom: Plane {
            normal: Vec3::NEG_Y,
            distance: 1.0,
        },
    };

    /// Computes the frustum of the given view-projection matrix.
    #[allow(dead_code)]
    pub fn new(view_proj: Mat4) -> Self {
        // x, y, z, w
        let mat = view_proj.to_cols_array_2d();

        // x
        let left = Plane::new(
            Vec3::new(
                mat[0][3] + mat[0][0],
                mat[1][3] + mat[1][0],
                mat[2][3] + mat[2][0],
            ),
            mat[3][3] + mat[3][0],
        );
        let right = Plane::new(
            Vec3::new(
                mat[0][3] - mat[0][0],
                mat[1][3] - mat[1][0],
                mat[2][3] - mat[2][0],
            ),
            mat[3][3] - mat[3][0],
        );

        // y
        let top = Plane::new(
            Vec3::new(
                mat[0][3] - mat[0][1],
                mat[1][3] - mat[1][1],
                mat[2][3] - mat[2][1],
            ),
            mat[3][3] - mat[3][1],
        );
        let bottom = Plane::new(
            Vec3::new(
                mat[0][3] + mat[0][1],
                mat[1][3] + mat[1][1],
                mat[2][3] + mat[2][1],
            ),
            mat[3][3] + mat[3][1],
        );

        // z (reversed)
        let near = Plane::new(
            Vec3::new(
                mat[0][3] - mat[0][2],
                mat[1][3] - mat[1][2],
                mat[2][3] - mat[2][2],
            ),
            mat[3][3] - mat[3][2],
        );

        // Normalize plane normals.
        Self {
            near: near.normalized(),
            left: left.normalized(),
            right: right.normalized(),
            top: top.normalized(),
            bottom: bottom.normalized(),
        }
    }

    /// Returns `true` if the given bounding sphere is inside the frustum.
    #[allow(dead_code)]
    pub fn contains_sphere(&self, sphere: &BoundingSphere) -> bool {
        let neg_radius = -sphere.radius;
        // Check if sphere is inside all planes (with normals pointing outside).
        self.near.distance_to_point(sphere.center) >= neg_radius
            && self.left.distance_to_point(sphere.center) >= neg_radius
            && self.right.distance_to_point(sphere.center) >= neg_radius
            && self.top.distance_to_point(sphere.center) >= neg_radius
            && self.bottom.distance_to_point(sphere.center) >= neg_radius
    }
}

/// Plane in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f32,
}

impl Plane {
    /// Creates a new plane from the given normal and distance.
    #[allow(dead_code)]
    pub fn new(normal: Vec3, distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Normalizes the plane.
    pub fn normalized(mut self) -> Self {
        let length = self.normal.length();
        self.normal /= length;
        self.distance /= length;
        self
    }

    /// Returns a signed distance from the plane to the given point.
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        // Project "origin to point" vector onto plane normal and add distance along normal.
        self.normal.dot(point) + self.distance
    }
}

impl gfx::AsStd140 for Plane {
    type Output = Vec4;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        Vec4::from(self)
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        *dst = Vec4::from(self);
    }
}

impl gfx::AsStd430 for Plane {
    type Output = Vec4;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        Vec4::from(self)
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        *dst = Vec4::from(self);
    }
}

impl From<Plane> for Vec4 {
    #[inline]
    fn from(value: Plane) -> Self {
        value.normal.extend(value.distance)
    }
}

impl From<&Plane> for Vec4 {
    #[inline]
    fn from(value: &Plane) -> Self {
        value.normal.extend(value.distance)
    }
}

/// Bounding sphere of a mesh.
#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

impl BoundingSphere {
    /// Computes the bounding sphere of the given list of positions.
    pub fn compute_from_positions(positions: &[Position]) -> Self {
        if positions.is_empty() {
            return Self {
                center: Vec3::ZERO,
                radius: 0.0,
            };
        }

        let center = positions.iter().fold(Vec3::ZERO, |acc, p| acc + p.0) / positions.len() as f32;
        let radius = positions
            .iter()
            .fold(0.0f32, |acc, p| acc.max((p.0 - center).length()));
        Self { center, radius }
    }

    /// Returns `true` if the given point is inside the bounding sphere.
    pub fn contains_point(&self, point: Vec3) -> bool {
        (point - self.center).length_squared() <= self.radius * self.radius
    }

    /// Transforms the bounding sphere by the given transform matrix.
    ///
    /// # Panics
    /// Panics if the transform matrix contains a perspective projection.
    pub fn transformed(self, transform: &Mat4) -> Self {
        let max_scale = transform
            .x_axis
            .xyz()
            .length_squared()
            .max(
                transform
                    .y_axis
                    .xyz()
                    .length_squared()
                    .max(transform.z_axis.xyz().length_squared()),
            )
            .sqrt();

        Self {
            center: transform.transform_point3(self.center),
            radius: self.radius * max_scale,
        }
    }
}

impl gfx::AsStd140 for BoundingSphere {
    type Output = Vec4;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        Vec4::from(self)
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        *dst = Vec4::from(self);
    }
}

impl gfx::AsStd430 for BoundingSphere {
    type Output = Vec4;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        Vec4::from(self)
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        *dst = Vec4::from(self);
    }
}

impl From<BoundingSphere> for Vec4 {
    #[inline]
    fn from(value: BoundingSphere) -> Self {
        value.center.extend(value.radius)
    }
}

impl From<&BoundingSphere> for Vec4 {
    #[inline]
    fn from(value: &BoundingSphere) -> Self {
        value.center.extend(value.radius)
    }
}
