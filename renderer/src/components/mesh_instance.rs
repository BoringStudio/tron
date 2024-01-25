use bevy_ecs::component::Component;

use crate::types::{MaterialHandle, MeshHandle};

#[derive(Debug, Clone, PartialEq, Component)]
pub struct MeshInstance2D {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

#[derive(Debug, Clone, PartialEq, Component)]
pub struct MeshInstance3D {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}
