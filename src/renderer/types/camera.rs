use glam::{Mat4, Vec3, Vec4};

pub struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    fovy: f32,
    znear: f32,

    projection: Mat4,
    view: Mat4,
}

impl Camera {
    pub fn new() -> Self {
        const FOVY: f32 = 75.0;
        const ZNEAR: f32 = 0.1;

        let mut camera = Self {
            eye: Vec3::new(0.0, 2.0, 3.0),
            target: Vec3::new(0.0, 1.0, 0.0),
            up: Vec3::Y,
            fovy: FOVY.to_radians(),
            znear: ZNEAR,
            projection: Mat4::IDENTITY,
            view: Mat4::ZERO,
        };
        camera.update_view_matrix();
        camera
    }

    pub fn update_projection(&mut self, aspect: f32) {
        self.projection = Mat4::perspective_infinite_reverse_rh(self.fovy, aspect, self.znear);
    }

    pub fn update_view_matrix(&mut self) {
        self.view = Mat4::look_at_rh(self.eye, self.target, self.up);
    }

    pub fn compute_view_proj(&self) -> glam::Mat4 {
        OPENGL_TO_WGPU_MATRIX * self.projection * self.view
    }
}

/// Map Z coords from -1..1 to 0..1
const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.5, 0.0),
    Vec4::new(0.0, 0.0, 0.5, 1.0),
);
