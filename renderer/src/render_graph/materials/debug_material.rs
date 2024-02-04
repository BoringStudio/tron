use anyhow::Result;
use glam::Vec3;

use crate::managers::GpuObject;
use crate::render_graph::render_passes::MainPass;
use crate::render_graph::{RenderGraphNode, RenderGraphNodeContext};
use crate::types::{MaterialInstance, Sorting, VertexAttributeArray, VertexAttributeKind};
use crate::util::{CachedGraphicsPipeline, RenderPassEncoderExt, ShaderPreprocessor};

pub struct DebugMaterial {
    pipeline: CachedGraphicsPipeline,
}

impl DebugMaterial {
    pub fn new(
        device: &gfx::Device,
        pipeline_layout: &gfx::PipelineLayout,
        shaders: &ShaderPreprocessor,
    ) -> Result<Self> {
        let shaders = shaders.begin();

        let vertex_shader = shaders.make_vertex_shader(device, "opaque_mesh.vert", "main")?;
        let fragment_shader = shaders.make_fragment_shader(device, "opaque_mesh.frag", "main")?;

        Ok(Self {
            pipeline: CachedGraphicsPipeline::new(gfx::GraphicsPipelineDescr {
                vertex_bindings: Vec::new(),
                vertex_attributes: Vec::new(),
                primitive_topology: Default::default(),
                primitive_restart_enable: false,
                vertex_shader,
                rasterizer: Some(gfx::Rasterizer {
                    fragment_shader: Some(fragment_shader),
                    front_face: gfx::FrontFace::CCW,
                    cull_mode: Some(gfx::CullMode::Back),
                    depth_test: Some(gfx::DepthTest {
                        compare: gfx::CompareOp::Less,
                        write: true,
                    }),
                    ..Default::default()
                }),
                layout: pipeline_layout.clone(),
            }),
        })
    }
}

impl RenderGraphNode for DebugMaterial {
    type RenderPass = MainPass;

    fn execute<'a, 'pass>(&mut self, ctx: &mut RenderGraphNodeContext<'a, 'pass>) -> Result<()> {
        let Some(material_instances_buffer) =
            ctx.synced_managers
                .material_manager
                .materials_data_buffer_handle::<DebugMaterialInstance>()
        else {
            return Ok(());
        };

        let frustum = &ctx.globals.frustum;

        ctx.encoder
            .bind_cached_graphics_pipeline(&mut self.pipeline, &ctx.state.device)?;

        if let Some(static_objects) = ctx
            .synced_managers
            .object_manager
            .iter_static_objects::<DebugMaterialInstance>()
        {
            ctx.encoder.push_constants(
                ctx.graphics_pipeline_layout,
                gfx::ShaderStageFlags::ALL,
                0,
                &[
                    ctx.state.mesh_manager.vertex_buffer_handle().index(),
                    static_objects.buffer_handle().index(),
                    material_instances_buffer.index(),
                ],
            );

            for (slot, object) in static_objects {
                // if !frustum.contains_sphere(&object.global_bounding_sphere) {
                //     continue;
                // }

                ctx.encoder.draw_indexed(
                    object.first_index..object.first_index + object.index_count,
                    0,
                    slot..slot + 1,
                );
            }
        }

        if let Some(dynamic_objects) = ctx
            .synced_managers
            .object_manager
            .iter_dynamic_objects::<DebugMaterialInstance>()
            .filter(|iter| iter.len() > 0)
        {
            let mut arena = ctx.state.multi_buffer_arena.begin::<DebugGpuObject>(
                &ctx.state.device,
                dynamic_objects.len(),
                gfx::BufferUsage::STORAGE,
            )?;

            // TODO: make it one iteration
            for object in dynamic_objects.clone() {
                arena.write(&object.as_interpolated_std430(ctx.interpolation_factor));
            }

            let objects_buffer_handle = ctx.state.multi_buffer_arena.end(
                &ctx.state.device,
                &ctx.state.bindless_resources,
                arena,
            );

            ctx.encoder.push_constants(
                ctx.graphics_pipeline_layout,
                gfx::ShaderStageFlags::ALL,
                0,
                &[
                    ctx.state.mesh_manager.vertex_buffer_handle().index(),
                    objects_buffer_handle.index(),
                    material_instances_buffer.index(),
                ],
            );

            for (slot, object) in dynamic_objects.enumerate() {
                ctx.encoder.draw_indexed(
                    object.first_index..object.first_index + object.index_count(),
                    0,
                    slot as u32..slot as u32 + 1,
                );
            }
        }

        Ok(())
    }
}

type DebugGpuObject = GpuObject<
    <<DebugMaterialInstance as MaterialInstance>::SupportedAttributes as VertexAttributeArray>::U32Array
>;

#[derive(Debug, Clone, Copy)]
pub struct DebugMaterialInstance {
    pub color: Vec3,
}

impl MaterialInstance for DebugMaterialInstance {
    type ShaderDataType = <Vec3 as gfx::AsStd430>::Output;
    type RequiredAttributes = [VertexAttributeKind; 1];
    type SupportedAttributes = [VertexAttributeKind; 5];

    fn required_attributes() -> Self::RequiredAttributes {
        [VertexAttributeKind::Position]
    }
    fn supported_attributes() -> Self::SupportedAttributes {
        [
            VertexAttributeKind::Position,
            VertexAttributeKind::Normal,
            VertexAttributeKind::Tangent,
            VertexAttributeKind::UV0,
            VertexAttributeKind::Color,
        ]
    }

    fn key(&self) -> u64 {
        0
    }

    fn sorting(&self) -> Sorting {
        Sorting::OPAQUE
    }

    fn shader_data(&self) -> Self::ShaderDataType {
        gfx::AsStd430::as_std430(&self.color)
    }
}
