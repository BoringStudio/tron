use glam::{Mat4, Vec3A};

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub projection: CameraProjection,
    pub view: Mat4,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            projection: CameraProjection::default(),
            view: Mat4::IDENTITY,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CameraProjection {
    Orhographic {
        /// Width, height and near depth of the orthographic view volume.
        extent: Vec3A,
    },
    Perspective {
        /// Vertical field of view in radians.
        fovy: f32,
        /// Near depth of the perspective view volume.
        near: f32,
    },
    Custom(Mat4),
}

impl CameraProjection {
    pub fn compute_projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        match self {
            Self::Orhographic { extent } => {
                let half = *extent * 0.5;
                Mat4::orthographic_rh(-half.x, half.x, -half.y, half.y, -half.z, half.z)
            }
            Self::Perspective { fovy, near } => {
                Mat4::perspective_infinite_reverse_rh(*fovy, aspect_ratio, *near)
            }
            Self::Custom(mat) => *mat,
        }
    }
}

impl Default for CameraProjection {
    fn default() -> Self {
        Self::Perspective {
            fovy: std::f32::consts::PI / 3.0,
            near: 0.1,
        }
    }
}
