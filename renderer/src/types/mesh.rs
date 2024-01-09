use crate::resource_handle::ResourceHandle;

pub struct Mesh {
    pub vertex_count: u32,
    pub positions: Vec<glam::Vec3>,
    pub normals: Vec<glam::Vec3>,
    pub tangents: Vec<glam::Vec3>,
    pub uv0: Vec<glam::Vec2>,

    pub indices: Vec<u32>,
}

pub type MeshHandle = ResourceHandle<Mesh>;

// TODO: add type for storing arbitrary vertex attributes
