use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use bevy_ecs::bundle::Bundle;
use bevy_ecs::world::World;
use glam::{Mat4, Vec2, Vec3};
use rand::Rng;

use ecs::components::Transform;
use renderer::components::StaticMeshInstance;
use renderer::{MeshHandle, RendererState};

pub struct Scene {
    pub ecs: World,
    pub primitive_meshes: PrimitiveMeshes,
}

impl Scene {
    pub fn new(renderer: &Arc<RendererState>) -> Result<Self> {
        Ok(Self {
            ecs: World::default(),
            primitive_meshes: PrimitiveMeshes::new(renderer)?,
        })
    }

    pub fn load_gltf<P: AsRef<Path>>(
        &mut self,
        path: P,
        renderer: &Arc<RendererState>,
    ) -> Result<()> {
        fn load_gltf_impl(
            path: &Path,
            ecs_world: &mut World,
            renderer: &Arc<RendererState>,
        ) -> Result<()> {
            let (gltf, buffers, _images) = gltf::import(path)?;
            let scene = gltf
                .default_scene()
                .context("default glTF scene not found")?;

            let mut stack = Vec::new();
            for node in scene.nodes() {
                stack.push((node.children(), Mat4::IDENTITY, Some(node)));

                while let Some((children, transform, node)) = stack.last_mut() {
                    if let Some(node) = node.take() {
                        process_gltf_node(node, &buffers, transform, ecs_world, renderer)?;
                    }

                    if let Some(child) = children.next() {
                        let child_transform = transform
                            .mul_mat4(&Mat4::from_cols_array_2d(&child.transform().matrix()));
                        stack.push((child.children(), child_transform, Some(child)));
                    } else {
                        stack.pop();
                    }
                }
            }

            Ok(())
        }

        load_gltf_impl(path.as_ref(), &mut self.ecs, renderer)
    }

    pub fn spawn_cube(&mut self, renderer: &Arc<RendererState>) -> Result<()> {
        let mut rng = rand::thread_rng();

        let material = renderer.add_material(renderer::DebugMaterial {
            color: glam::vec3(
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ),
        });

        let transform = Transform::from_translation(glam::vec3(
            rng.gen_range(-5.0..5.0),
            -1.0,
            rng.gen_range(-5.0..5.0),
        ))
        .with_scale(Vec3::splat(rng.gen_range(0.1..0.5)));

        let handle = renderer.add_static_object(renderer::StaticObject {
            mesh: self.primitive_meshes.cube.clone(),
            material: material.clone(),
            transform: transform.to_matrix(),
        });

        self.ecs.spawn(SceneObjectBundle {
            transform,
            mesh_instance: StaticMeshInstance {
                mesh: self.primitive_meshes.cube.clone(),
                material,
                handle,
            },
        });

        Ok(())
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

fn process_gltf_node(
    node: gltf::Node,
    buffers: &[gltf::buffer::Data],
    global_transform: &Mat4,
    ecs_world: &mut World,
    renderer: &Arc<RendererState>,
) -> Result<()> {
    let Some(mesh) = node.mesh() else {
        return Ok(());
    };

    for primitive in mesh.primitives() {
        let reader =
            primitive.reader(|buffer| buffers.get(buffer.index()).map(std::ops::Deref::deref));
        let Some(positions) = reader.read_positions() else {
            continue;
        };
        let Some(indices) = reader.read_indices() else {
            continue;
        };

        let vertex_count = positions.len();

        #[inline]
        fn optional_iter<I, T: Default>(iter: Option<I>, len: usize) -> Result<Option<I>>
        where
            I: Iterator<Item = T> + ExactSizeIterator,
        {
            if let Some(iter) = &iter {
                anyhow::ensure!(iter.len() == len, "component array length mismatch");
            }
            Ok(iter)
        }

        let normals = optional_iter(reader.read_normals(), vertex_count)?;
        let tangents = optional_iter(reader.read_tangents(), vertex_count)?;
        let uv0 = optional_iter(
            reader.read_tex_coords(0).map(|iter| iter.into_f32()),
            vertex_count,
        )?;

        let mesh = {
            let mut builder = renderer::Mesh::builder(
                positions
                    .map(|[x, y, z]| renderer::Position(Vec3::new(x, y, z)))
                    .collect::<Vec<_>>(),
            );

            if let Some(normals) = normals {
                builder = builder.with_normals(
                    normals
                        .map(|[x, y, z]| renderer::Normal(Vec3::new(x, y, z)))
                        .collect::<Vec<_>>(),
                );
            } else {
                builder = builder.with_computed_normals();
            }

            if let Some(tangents) = tangents {
                builder = builder.with_tangents(
                    tangents
                        .map(|[x, y, z, _]| renderer::Tangent(Vec3::new(x, y, z)))
                        .collect::<Vec<_>>(),
                );
            }
            if let Some(uv0) = uv0 {
                builder = builder.with_uv0(
                    uv0.map(|[x, y]| renderer::UV0(Vec2::new(x, y)))
                        .collect::<Vec<_>>(),
                );
            }

            builder.with_indices(indices.into_u32().collect()).build()?
        };

        let mesh = renderer.add_mesh(&mesh)?;
        let material = renderer.add_material(renderer::DebugMaterial {
            color: glam::vec3(1.0, 1.0, 1.0),
        });

        let handle = renderer.add_static_object(renderer::StaticObject {
            mesh: mesh.clone(),
            material: material.clone(),
            transform: *global_transform,
        });

        ecs_world.spawn(SceneObjectBundle {
            transform: Transform::from_matrix(*global_transform),
            mesh_instance: StaticMeshInstance {
                mesh,
                material,
                handle,
            },
        });
    }

    Ok(())
}

#[derive(Bundle)]
struct SceneObjectBundle {
    transform: Transform,
    mesh_instance: StaticMeshInstance,
}
