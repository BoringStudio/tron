pub use self::camera::Camera;
pub use self::mesh::{Mesh, Vertex};
pub use self::texture::Texture;

use crate::util::{RawResourceHandle, ResourceHandle};

pub mod camera;
pub mod mesh;
pub mod texture;

pub type MeshHandle = ResourceHandle<Mesh>;
pub type RawMeshHandle = RawResourceHandle<Mesh>;
