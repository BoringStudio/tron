use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use bevy_ecs::system::Resource;
use renderer::{MeshHandle, RendererState};

#[derive(Resource)]
pub struct Time {
    pub started_at: Instant,
    pub now: Instant,
    pub step: Duration,
}

#[derive(Resource)]
pub struct Graphics {
    pub renderer: Arc<RendererState>,
    pub primitive_meshes: PrimitiveMeshes,
}

impl Graphics {
    pub fn new(renderer: Arc<RendererState>) -> Result<Self> {
        let primitive_meshes = PrimitiveMeshes::new(&renderer)?;

        Ok(Self {
            renderer,
            primitive_meshes,
        })
    }
}

pub struct PrimitiveMeshes {
    pub cube: MeshHandle,
    pub plane: MeshHandle,
}

impl PrimitiveMeshes {
    pub fn new(state: &Arc<RendererState>) -> Result<Self> {
        let cube = state.add_mesh(
            &renderer::Mesh::builder(renderer::CubeMeshGenerator::from_size(1.0))
                .with_computed_normals()
                .build()?,
        )?;

        let plane = state.add_mesh(
            &renderer::Mesh::builder(renderer::PlaneMeshGenerator::from_size(1.0))
                .with_computed_normals()
                .build()?,
        )?;

        Ok(Self { cube, plane })
    }
}
