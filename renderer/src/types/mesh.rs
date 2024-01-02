use crate::resource_registry::ResourceHandle;

pub struct Mesh {
    pub vertex_data: Vec<u8>,
    pub vertex_size: usize,
    pub indices: Vec<u32>,
}

pub type MeshHandle = ResourceHandle<Mesh>;
