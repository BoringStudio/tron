use glam::Mat4;

use crate::types::{MaterialInstanceHandle, MeshHandle};
use crate::util::{RawResourceHandle, ResourceHandle};

pub type StaticObjectHandle = ResourceHandle<StaticObjectTag>;
pub(crate) type RawStaticObjectHandle = RawResourceHandle<StaticObjectTag>;

pub type DynamicObjectHandle = ResourceHandle<DynamicObjectTag>;
pub(crate) type RawDynamicObjectHandle = RawResourceHandle<DynamicObjectTag>;

pub struct StaticObjectTag;
pub struct DynamicObjectTag;

pub struct ObjectData {
    pub mesh: MeshHandle,
    pub material: MaterialInstanceHandle,
    pub global_transform: Mat4,
}
