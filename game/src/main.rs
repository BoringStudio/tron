use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use argh::FromArgs;
use bevy_ecs::prelude::*;
use ecs::components::Transform;
use renderer::components::{DynamicMeshInstance, StaticMeshInstance};
use winit::event::*;
use winit::event_loop::EventLoopBuilder;
#[cfg(wayland_platform)]
use winit::platform::wayland::EventLoopBuilderExtWayland;
#[cfg(x11_platform)]
use winit::platform::x11::{EventLoopBuilderExtX11, WindowBuilderExtX11, XWindowType};
use winit::window::WindowBuilder;

use renderer::Renderer;

use self::scene::Scene;

mod scene;

#[cfg(not(any(target_env = "msvc", miri)))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> Result<()> {
    let app: App = argh::from_env();
    app.run()
}

/// Vulkan rendering experiments
#[derive(FromArgs)]
struct App {
    // TEMP
    /// glTF file to load
    #[argh(positional)]
    gltf_scene: Option<String>,

    /// enable profiling server
    #[argh(switch)]
    profiling: bool,

    /// use Vulkan validation layer
    #[argh(switch)]
    vk_validation_layer: bool,

    /// generate shader debug info
    #[argh(switch)]
    vk_debug_shaders: bool,

    /// enable X11-specific popup mode
    #[cfg(x11_platform)]
    #[argh(switch)]
    x11_as_popup: bool,

    /// force use X11 window backend
    #[cfg(x11_platform)]
    #[argh(switch)]
    x11_backend: bool,

    /// force use Wayland window backend
    #[cfg(wayland_platform)]
    #[argh(switch)]
    wayland_backend: bool,
}

impl App {
    pub fn run(self) -> Result<()> {
        #[cfg(all(x11_platform, wayland_platform))]
        if self.x11_backend && self.wayland_backend {
            panic!("can't use both X11 and Wayland backends");
        }

        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::builder()
                    .with_default_directive(tracing::Level::INFO.into())
                    .from_env_lossy(),
            )
            .init();

        let _puffin_server = if self.profiling {
            let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
            let puffin_server = puffin_http::Server::new(&server_addr).unwrap();
            tracing::info!(server_addr, "started profiling server");
            Some(puffin_server)
        } else {
            None
        };
        profiling::puffin::set_scopes_on(self.profiling);

        let app_name = env!("CARGO_BIN_NAME").to_owned();

        let event_loop = {
            let mut builder = EventLoopBuilder::new();

            #[cfg(x11_platform)]
            if self.x11_backend {
                builder.with_x11();
            }
            #[cfg(wayland_platform)]
            if self.wayland_backend {
                builder.with_wayland();
            }

            builder.build()?
        };

        let window = {
            let mut builder = WindowBuilder::new();
            builder = builder.with_title(app_name);

            #[cfg(x11_platform)]
            if self.x11_as_popup {
                builder =
                    builder.with_x11_window_type(vec![XWindowType::Dialog, XWindowType::Normal]);
            }

            builder.build(&event_loop).map(Arc::new)?
        };

        let started_at = Instant::now();

        let mut renderer = Renderer::builder(window.clone(), started_at)
            .app_version((0, 0, 1))
            .validation_layer(self.vk_validation_layer)
            .shaders_debug_info_enabled(self.vk_debug_shaders)
            .build()?;

        let mut scene = Scene::new(renderer.state())?;
        scene.ecs.insert_resource(Time {
            started_at,
            now: started_at,
            step: Duration::from_secs(1) / 10,
        });
        scene.ecs.insert_resource(RendererResources {
            state: renderer.state().clone(),
        });

        if let Some(gltf_scene_path) = self.gltf_scene {
            scene.load_gltf(&gltf_scene_path, renderer.state())?;
        }

        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                rotation_system,
                apply_static_objects_transform_system,
                apply_dynamic_objects_transform_system,
                sync_fixed_update_system,
            )
                .chain(),
        );

        let mut minimized = false;
        let handle_event = {
            let renderer_state = renderer.state().clone();
            move |event, elwt: &winit::event_loop::EventLoopWindowTarget<_>| {
                elwt.set_control_flow(winit::event_loop::ControlFlow::Poll);

                {
                    let now = Instant::now();

                    let (mut updated_at, step) = {
                        let time = scene.ecs.resource::<Time>();
                        (time.now, time.step)
                    };
                    loop {
                        updated_at += step;
                        if updated_at > now {
                            break;
                        }

                        scene.ecs.resource_mut::<Time>().now = updated_at;
                        schedule.run(&mut scene.ecs);
                    }
                }

                match event {
                    winit::event::Event::AboutToWait => window.request_redraw(),
                    winit::event::Event::WindowEvent { event, .. } => match event {
                        WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                            renderer_state.notify_draw();
                        }
                        WindowEvent::Resized(size) => {
                            minimized = size.width == 0 || size.height == 0;
                            renderer_state.notify_resized();
                        }
                        WindowEvent::CloseRequested => {
                            renderer_state.set_running(false);
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
                                    if let Err(e) = scene.spawn_cube(&renderer_state) {
                                        tracing::error!("failed to add test object: {}", e);
                                    } else {
                                        tracing::info!("added test object");
                                    }
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        };

        tracing::debug!("event loop started");
        event_loop.run(handle_event)?;
        tracing::debug!("event loop stopped");

        renderer.cleanup()?;

        Ok(())
    }
}

#[derive(Resource)]
struct Time {
    #[allow(unused)]
    started_at: Instant,
    now: Instant,
    step: Duration,
}

#[derive(Resource)]
struct RendererResources {
    state: Arc<renderer::RendererState>,
}

fn rotation_system(res: Res<Time>, mut query: Query<&mut Transform>) {
    for mut transform in &mut query {
        transform.rotate_y(1.0 * res.step.as_secs_f32());
    }
}

fn apply_static_objects_transform_system(
    res: Res<RendererResources>,
    query: Query<(&Transform, &StaticMeshInstance, Changed<Transform>)>,
) {
    for (transform, object, _) in &query {
        res.state
            .update_static_object(&object.handle, transform.to_matrix());
    }
}

fn apply_dynamic_objects_transform_system(
    res: Res<RendererResources>,
    query: Query<(&Transform, &DynamicMeshInstance, Changed<Transform>)>,
) {
    for (transform, object, _) in &query {
        res.state
            .update_dynamic_object(&object.handle, transform.to_matrix(), false);
    }
}

fn sync_fixed_update_system(time: Res<Time>, renderer: Res<RendererResources>) {
    renderer.state.finish_fixed_update(time.now, time.step);
}
