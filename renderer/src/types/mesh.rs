use crate::resource_handle::ResourceHandle;
use crate::types::VertexAttributeData;

pub struct Mesh {
    pub vertex_count: u32,
    pub attribute_data: Vec<VertexAttributeData>,
    pub indices: Vec<u32>,
}

pub type MeshHandle = ResourceHandle<Mesh>;
