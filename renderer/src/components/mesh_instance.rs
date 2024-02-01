use bevy_ecs::component::Component;

use crate::types::{DynamicObjectHandle, MaterialInstanceHandle, MeshHandle, StaticObjectHandle};

#[derive(Debug, Clone, PartialEq, Component)]
pub struct StaticMeshInstance {
    pub mesh: MeshHandle,
    pub material: MaterialInstanceHandle,
    pub handle: StaticObjectHandle,
}

#[derive(Debug, Clone, PartialEq, Component)]
pub struct DynamicMeshInstance {
    pub mesh: MeshHandle,
    pub material: MaterialInstanceHandle,
    pub handle: DynamicObjectHandle,
}
