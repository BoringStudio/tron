use glam::Mat4;

use crate::types::{MaterialHandle, MeshHandle};
use crate::util::{RawResourceHandle, ResourceHandle};

pub type StaticObjectHandle = ResourceHandle<StaticObject>;
pub(crate) type RawStaticObjectHandle = RawResourceHandle<StaticObject>;

#[derive(Debug, Clone, PartialEq)]
pub struct StaticObject {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    pub transform: Mat4,
}
