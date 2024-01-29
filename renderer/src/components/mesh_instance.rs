use bevy_ecs::component::Component;

use crate::types::{MaterialHandle, MeshHandle, StaticObjectHandle};

/// A mesh instance data is prepared in advance and explicitly
/// managed though the renderer state.
#[derive(Debug, Clone, PartialEq, Component)]
pub struct StaticMeshInstance {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub handle: StaticObjectHandle,
}

/// A mesh instance data is collected on each frame.
#[derive(Debug, Clone, PartialEq, Component)]
pub struct DynamicMeshInstance {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}
