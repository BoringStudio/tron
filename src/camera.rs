pub struct Camera {
    eye: glm::Vec3,
    target: glm::Vec3,
    up: glm::Vec3,
    fovy: f32,
    znear: f32,
    zfar: f32,

    projection: glm::Mat4,
    view: glm::Mat4,
}

impl Camera {
    pub fn new() -> Self {
        const FOVY: f32 = 75.0;
        const ZRANGE: (f32, f32) = (0.1, 100.0);

        let mut camera = Self {
            eye: glm::vec3(0.0, 1.0, 2.0),
            target: glm::vec3(0.0, 0.0, 0.0),
            up: glm::Vec3::y(),
            fovy: FOVY.to_radians(),
            znear: ZRANGE.0,
            zfar: ZRANGE.1,
            projection: glm::identity(),
            view: Default::default(),
        };
        camera.update_view_matrix();
        camera
    }

    pub fn update_projection(&mut self, aspect: f32) {
        self.projection = glm::perspective(aspect, self.fovy, self.znear, self.zfar);
    }

    pub fn update_view_matrix(&mut self) {
        self.view = glm::look_at_rh(&self.eye, &self.target, &self.up);
    }

    pub fn compute_view_proj(&self) -> glm::Mat4 {
        OPENGL_TO_WGPU_MATRIX * self.projection * self.view
    }
}

const OPENGL_TO_WGPU_MATRIX: glm::Mat4 = glm::Mat4::new(
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 0.5, 0.0, //
    0.0, 0.0, 0.5, 1.0, //
);
