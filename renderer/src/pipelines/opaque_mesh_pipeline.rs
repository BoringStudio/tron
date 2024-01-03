use anyhow::Result;

use crate::shader_preprocessor::ShaderPreprocessor;

pub struct OpaqueMeshPipeline;

impl OpaqueMeshPipeline {
    pub fn make_descr(
        device: &gfx::Device,
        shaders: &mut ShaderPreprocessor,
    ) -> Result<gfx::GraphicsPipelineDescr> {
        let layout = device.create_pipeline_layout(gfx::PipelineLayoutInfo {
            sets: Default::default(),
            push_constants: Default::default(),
        })?;

        let shaders = shaders.begin();

        let vertex_shader = shaders.make_vertex_shader(device, "triangle.vert", "main")?;
        let fragment_shader = shaders.make_fragment_shader(device, "triangle.frag", "main")?;

        Ok(gfx::GraphicsPipelineDescr {
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            primitive_topology: Default::default(),
            primitive_restart_enable: false,
            vertex_shader,
            rasterizer: Some(gfx::Rasterizer {
                fragment_shader: Some(fragment_shader),
                ..Default::default()
            }),
            layout,
        })
    }
}
