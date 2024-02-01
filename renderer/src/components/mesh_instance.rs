use bevy_ecs::component::Component;

use crate::types::{DynamicObjectHandle, MaterialHandle, MeshHandle, StaticObjectHandle};

#[derive(Debug, Clone, PartialEq, Component)]
pub struct StaticMeshInstance {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub handle: StaticObjectHandle,
}

#[derive(Debug, Clone, PartialEq, Component)]
pub struct DynamicMeshInstance {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub handle: DynamicObjectHandle,
}
