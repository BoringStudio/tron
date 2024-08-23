use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ScheduleLabel;
use ecs::components::Transform;
use glam::{Mat4, Vec2, Vec3};
use rand::Rng;
use renderer::materials::DebugMaterialInstance;
use renderer::RendererState;
use winit::event::WindowEvent;

use self::components::{Camera, DynamicMeshInstance, StaticMeshInstance};
use self::resources::{Graphics, MainCamera, Time};

mod components;
mod resources;

pub struct Game {
    world: World,
    fixed_update_schedule: Schedule,
    draw_schedule: Schedule,
    minimized: bool,
}

impl Game {
    pub fn new(renderer: Arc<RendererState>) -> Result<Self> {
        let started_at = Instant::now();

        let mut world = World::default();
        world.insert_resource(Time {
            started_at,
            now: started_at,
            step: Duration::from_secs(1) / 10, // TEMP 10 FPS
        });
        world.insert_resource(MainCamera { entity: None });
        world.insert_resource(Graphics::new(renderer)?);

        let mut fixed_update_schedule = FixedUpdateSchedule::base_schedule();
        fixed_update_schedule.add_systems(rotate_objects_system.in_set(FixedUpdateSet::OnUpdate));
        fixed_update_schedule.add_systems(
            (
                (
                    apply_static_objects_transform_system,
                    apply_dynamic_objects_transform_system,
                ),
                sync_fixed_update_system,
            )
                .chain()
                .in_set(FixedUpdateSet::AfterUpdate),
        );

        let mut draw_schedule = DrawSchedule::base_schedule();
        draw_schedule.add_systems(apply_camera_transform_system.in_set(DrawSet::AfterDraw));

        let entity = world
            .spawn((
                Camera {
                    projection: Default::default(),
                },
                Transform::from_translation(Vec3::new(0.0, 0.5, 3.0))
                    .looking_at(Vec3::ZERO, Vec3::Y),
            ))
            .id();
        world.resource_mut::<MainCamera>().entity = Some(entity);

        Ok(Self {
            world,
            fixed_update_schedule,
            draw_schedule,
            minimized: false,
        })
    }

    pub fn handle_event(
        &mut self,
        event: winit::event::Event<()>,
        elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    ) {
        elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

        let mut redraw_requested = false;
        match event {
            winit::event::Event::AboutToWait => {
                self.world
                    .resource::<Graphics>()
                    .renderer
                    .window()
                    .request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested if !elwt.exiting() && !self.minimized => {
                    redraw_requested = true;
                }
                WindowEvent::Resized(size) => {
                    self.minimized = size.width == 0 || size.height == 0;
                }
                WindowEvent::CloseRequested => {
                    self.world
                        .resource::<Graphics>()
                        .renderer
                        .set_running(false);
                    elwt.exit();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    use winit::keyboard::{KeyCode, PhysicalKey};

                    let code = match event.physical_key {
                        PhysicalKey::Code(code) if event.state.is_pressed() => code,
                        _ => return,
                    };

                    match code {
                        KeyCode::ArrowRight => {
                            self.spawn_cube();
                            tracing::info!("added test object");
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            _ => {}
        }

        let now = Instant::now();

        let (mut updated_at, step) = {
            let time = self.world.resource::<Time>();
            (time.now, time.step)
        };
        loop {
            updated_at += step;
            if updated_at > now {
                break;
            }

            self.world.resource_mut::<Time>().now = updated_at;
            self.fixed_update_schedule.run(&mut self.world);
        }

        if redraw_requested {
            self.draw_schedule.run(&mut self.world);
            self.world.resource::<Graphics>().renderer.notify_draw();
        }
    }

    // TEMP
    pub fn load_gltf(&mut self, path: &Path) -> Result<()> {
        let (gltf, buffers, _images) = gltf::import(path)?;
        let scene = gltf
            .default_scene()
            .context("default glTF scene not found")?;

        let renderer = self.world.resource::<Graphics>().renderer.clone();

        let mut stack = Vec::new();
        for node in scene.nodes() {
            stack.push((node.children(), Mat4::IDENTITY, Some(node)));

            while let Some((children, transform, node)) = stack.last_mut() {
                if let Some(node) = node.take() {
                    process_gltf_node(node, &buffers, transform, &mut self.world, &renderer)?;
                }

                if let Some(child) = children.next() {
                    let child_transform =
                        transform.mul_mat4(&Mat4::from_cols_array_2d(&child.transform().matrix()));
                    stack.push((child.children(), child_transform, Some(child)));
                } else {
                    stack.pop();
                }
            }
        }

        Ok(())
    }

    // TEMP
    pub fn spawn_cube(&mut self) {
        let graphics = self.world.resource::<Graphics>();

        let mut rng = rand::thread_rng();

        let transform = Transform::from_translation(Vec3::new(
            rng.gen_range(-5.0..5.0),
            -1.0,
            rng.gen_range(-5.0..5.0),
        ))
        .with_scale(Vec3::splat(rng.gen_range(0.1..0.5)));

        let mesh = graphics.primitive_meshes.cube.clone();

        let material = graphics
            .renderer
            .add_material_instance(DebugMaterialInstance {
                color: Vec3::new(
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                ),
            });

        let handle = graphics.renderer.add_dynamic_object(
            mesh.clone(),
            material.clone(),
            &transform.to_matrix(),
        );

        self.world.spawn(SceneObjectBundle {
            transform,
            mesh_instance: DynamicMeshInstance {
                mesh,
                material,
                handle,
            },
        });
    }
}

#[derive(Debug, ScheduleLabel, Hash, PartialEq, Eq, Clone)]
pub struct FixedUpdateSchedule;

impl FixedUpdateSchedule {
    fn base_schedule() -> Schedule {
        let mut schedule = Schedule::new(Self);
        schedule.configure_sets(
            (
                FixedUpdateSet::BeforeUpdate,
                FixedUpdateSet::OnUpdate,
                FixedUpdateSet::AfterUpdate,
            )
                .chain(),
        );
        schedule
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum FixedUpdateSet {
    BeforeUpdate,
    OnUpdate,
    AfterUpdate,
}

#[derive(Debug, ScheduleLabel, Hash, PartialEq, Eq, Clone)]
pub struct DrawSchedule;

impl DrawSchedule {
    fn base_schedule() -> Schedule {
        let mut schedule = Schedule::new(Self);
        schedule.configure_sets((DrawSet::BeforeDraw, DrawSet::OnDraw, DrawSet::AfterDraw).chain());
        schedule
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum DrawSet {
    BeforeDraw,
    OnDraw,
    AfterDraw,
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
        let material = renderer.add_material_instance(renderer::materials::DebugMaterialInstance {
            color: glam::vec3(1.0, 1.0, 1.0),
        });

        let handle = renderer.add_dynamic_object(mesh.clone(), material.clone(), global_transform);

        ecs_world.spawn(SceneObjectBundle {
            transform: Transform::from_matrix(*global_transform),
            mesh_instance: DynamicMeshInstance {
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
    mesh_instance: DynamicMeshInstance,
}

// TEMP
fn rotate_objects_system(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &DynamicMeshInstance)>,
) {
    for (mut transform, _) in &mut query {
        transform.rotate_y(time.step.as_secs_f32());
    }
}

fn apply_static_objects_transform_system(
    graphics: Res<Graphics>,
    query: Query<(&Transform, &StaticMeshInstance), Changed<Transform>>,
) {
    for (transform, object) in &query {
        graphics
            .renderer
            .update_static_object(&object.handle, transform.to_matrix());
    }
}

fn apply_dynamic_objects_transform_system(
    graphics: Res<Graphics>,
    query: Query<(&Transform, &DynamicMeshInstance), Changed<Transform>>,
) {
    for (transform, object) in &query {
        graphics
            .renderer
            .update_dynamic_object(&object.handle, transform.to_matrix(), false);
    }
}

fn sync_fixed_update_system(time: Res<Time>, graphics: Res<Graphics>) {
    graphics.renderer.finish_fixed_update(time.now, time.step);
}

fn apply_camera_transform_system(
    graphics: Res<Graphics>,
    main_camera: Res<MainCamera>,
    world: &World,
) {
    let Some(entity) = main_camera.entity else {
        return;
    };

    let (Some(transform), Some(camera)) =
        (world.get::<Transform>(entity), world.get::<Camera>(entity))
    else {
        return;
    };

    graphics
        .renderer
        .update_camera(&transform.to_matrix().inverse(), &camera.projection);
}
