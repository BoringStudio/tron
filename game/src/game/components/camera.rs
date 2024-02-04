use bevy_ecs::component::Component;

use renderer::CameraProjection;

#[derive(Debug, Default, Clone, Copy, PartialEq, Component)]
pub struct Camera {
    pub projection: CameraProjection,
}
