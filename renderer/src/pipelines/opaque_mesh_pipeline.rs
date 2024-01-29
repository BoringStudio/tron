use anyhow::Result;

use crate::util::{BindlessResources, FrameResources, ShaderPreprocessor};

pub struct OpaqueMeshPipeline;

impl OpaqueMeshPipeline {
    pub fn make_descr(
        device: &gfx::Device,
        shaders: &ShaderPreprocessor,
        frame_resources: &FrameResources,
        bindless_resources: &BindlessResources,
    ) -> Result<gfx::GraphicsPipelineDescr> {
        let layout = device.create_pipeline_layout(gfx::PipelineLayoutInfo {
            sets: vec![
                frame_resources.descriptor_set_layout().clone(),
                bindless_resources.descriptor_set_layout().clone(),
            ],
            push_constants: vec![gfx::PushConstant {
                stages: gfx::ShaderStageFlags::ALL,
                offset: 0,
                size: 12,
            }],
        })?;

        let shaders = shaders.begin();

        let vertex_shader = shaders.make_vertex_shader(device, "opaque_mesh.vert", "main")?;
        let fragment_shader = shaders.make_fragment_shader(device, "opaque_mesh.frag", "main")?;

        Ok(gfx::GraphicsPipelineDescr {
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            primitive_topology: Default::default(),
            primitive_restart_enable: false,
            vertex_shader,
            rasterizer: Some(gfx::Rasterizer {
                fragment_shader: Some(fragment_shader),
                front_face: gfx::FrontFace::CW,
                cull_mode: Some(gfx::CullMode::Back),
                depth_test: Some(gfx::DepthTest {
                    compare: gfx::CompareOp::Less,
                    write: true,
                }),
                ..Default::default()
            }),
            layout,
        })
    }
}
