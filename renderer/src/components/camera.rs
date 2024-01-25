use bevy_ecs::component::Component;

use crate::types::CameraProjection;

#[derive(Debug, Default, Clone, Copy, PartialEq, Component)]
pub struct Camera {
    pub projection: CameraProjection,
}
