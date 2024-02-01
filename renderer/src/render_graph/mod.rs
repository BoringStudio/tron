use std::time::Instant;

use anyhow::Result;

use crate::render_graph::render_passes::MainPassInput;
use crate::util::{EncoderExt, RenderPass};
use crate::{RendererState, RendererStateSyncedManagers};

pub mod materials {
    pub use self::debug_material::{DebugMaterial, DebugMaterialInstance};

    mod debug_material;
}

mod render_passes {
    pub use self::main_pass::{MainPass, MainPassInput};

    mod main_pass;
}

// NOTE: This is a "fixed-function" stub for now.
pub struct RenderGraph {
    graphics_pipeline_layout: gfx::PipelineLayout,

    // TEMP
    main_pass: render_passes::MainPass,
    debug_material: materials::DebugMaterial,
}

impl RenderGraph {
    pub fn new(state: &RendererState) -> Result<Self> {
        let graphics_pipeline_layout =
            state
                .device
                .create_pipeline_layout(gfx::PipelineLayoutInfo {
                    sets: vec![
                        state.frame_resources.descriptor_set_layout().clone(),
                        state.bindless_resources.descriptor_set_layout().clone(),
                    ],
                    push_constants: vec![gfx::PushConstant {
                        stages: gfx::ShaderStageFlags::ALL,
                        offset: 0,
                        size: 12,
                    }],
                })?;

        let main_pass = render_passes::MainPass::default();
        let debug_material = materials::DebugMaterial::new(
            &state.device,
            &graphics_pipeline_layout,
            &state.shader_preprocessor,
        )?;

        Ok(Self {
            graphics_pipeline_layout,
            main_pass,
            debug_material,
        })
    }

    pub fn execute(&mut self, ctx: &mut RenderGraphContext<'_>) -> Result<()> {
        profiling::scope!("render_graph");

        let interpolation_factor = ctx
            .synced_managers
            .time_manager
            .compute_interpolation_factor(ctx.now);

        let globals_dynamic_offset = ctx.state.frame_resources.flush(ctx.delta_time, ctx.frame);

        ctx.encoder.bind_graphics_descriptor_sets(
            &self.graphics_pipeline_layout,
            0,
            &[
                ctx.state.frame_resources.descriptor_set(),
                ctx.state.bindless_resources.descriptor_set(),
            ],
            &[globals_dynamic_offset],
        );

        ctx.state.mesh_manager.bind_index_buffer(ctx.encoder);

        ctx.encoder.memory_barrier(
            gfx::PipelineStageFlags::COMPUTE_SHADER | gfx::PipelineStageFlags::TRANSFER,
            gfx::AccessFlags::SHADER_WRITE | gfx::AccessFlags::TRANSFER_WRITE,
            gfx::PipelineStageFlags::VERTEX_SHADER,
            gfx::AccessFlags::SHADER_READ,
        );

        {
            profiling::scope!("main_pass");

            let mut encoder = ctx.encoder.with_render_pass(
                &mut self.main_pass,
                &MainPassInput {
                    max_image_count: ctx.surface_image.total_image_count(),
                    target: ctx.surface_image.image().clone(),
                },
                &ctx.state.device,
            )?;

            self.debug_material.execute(&mut RenderGraphNodeContext {
                graphics_pipeline_layout: &self.graphics_pipeline_layout,
                state: ctx.state,
                synced_managers: ctx.synced_managers,
                encoder,
                now: ctx.now,
                delta_time: ctx.delta_time,
                frame: ctx.frame,
                interpolation_factor,
            })?;
        }

        Ok(())
    }
}

pub struct RenderGraphContext<'a> {
    pub state: &'a RendererState,
    pub synced_managers: &'a RendererStateSyncedManagers,
    pub surface_image: &'a gfx::SurfaceImage<'a>,
    pub encoder: &'a mut gfx::Encoder,
    pub now: Instant,
    pub delta_time: f32,
    pub frame: u32,
}

trait RenderGraphNode {
    type RenderPass: RenderPass;

    fn execute<'a, 'pass>(&mut self, ctx: &mut RenderGraphNodeContext<'a, 'pass>) -> Result<()>;
}

struct RenderGraphNodeContext<'a, 'pass> {
    pub graphics_pipeline_layout: &'a gfx::PipelineLayout,
    pub state: &'a RendererState,
    pub synced_managers: &'a RendererStateSyncedManagers,
    pub encoder: gfx::RenderPassEncoder<'a, 'pass>,
    pub now: Instant,
    pub delta_time: f32,
    pub frame: u32,
    pub interpolation_factor: f32,
}
