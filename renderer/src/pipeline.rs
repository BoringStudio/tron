use std::rc::Rc;

use anyhow::Result;
use glam::{Vec2, Vec3};
use vulkanalia::prelude::v1_0::*;

use super::base::RendererBase;
use super::pipeline_layout::PipelineLayout;
use super::shader_module::ShaderModule;

#[derive(Debug, Clone)]
pub struct SurfaceDescription {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

pub struct Pipeline {
    base: Rc<RendererBase>,
    pipeline_layout: PipelineLayout,
    render_pass: SimpleRenderPass,
    handle: vk::Pipeline,
}

impl Pipeline {
    pub unsafe fn new(base: Rc<RendererBase>, surface: &SurfaceDescription) -> Result<Self> {
        let mesh_shader = ShaderModule::new(base.clone(), SIMPLE_SHADER)?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(mesh_shader.handle())
            .name(b"vs_main\0");
        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(mesh_shader.handle())
            .name(b"fs_main\0");

        let binding_descriptions = &[Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(surface.extent);
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(render_area.extent.width as f32)
            .height(render_area.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&render_area));

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::_1);

        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(std::slice::from_ref(&attachment))
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        // let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        // let dynamic_state =
        //     vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

        // TODO: use builders
        let pipeline_layout = PipelineLayout::new(base.clone())?;
        let render_pass = SimpleRenderPass::new(base.clone(), surface)?;

        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            //.dynamic_state(&dynamic_state)
            .layout(pipeline_layout.handle())
            .render_pass(render_pass.handle())
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        let handle = base
            .device()
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&info),
                None,
            )?
            .0[0];

        Ok(Pipeline {
            base,
            pipeline_layout,
            render_pass,
            handle,
        })
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    pub fn render_pass(&self) -> &SimpleRenderPass {
        &self.render_pass
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.base.device().destroy_pipeline(self.handle, None);
        }
    }
}

pub struct SimpleRenderPass {
    base: Rc<RendererBase>,
    handle: vk::RenderPass,
}

impl SimpleRenderPass {
    unsafe fn new(base: Rc<RendererBase>, surface: &SurfaceDescription) -> Result<Self> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(surface.format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        let handle = base.device().create_render_pass(&render_pass_info, None)?;

        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::RenderPass {
        self.handle
    }
}

impl Drop for SimpleRenderPass {
    fn drop(&mut self) {
        unsafe {
            self.base.device().destroy_render_pass(self.handle, None);
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    position: Vec2,
    color: Vec3,
}

impl Vertex {
    const fn new(position: Vec2, color: Vec3) -> Self {
        Self { position, color }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let position = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::size_of::<Vec2> as u32)
            .build();
        [position, color]
    }
}

const VERTICES: [Vertex; 3] = [
    Vertex::new(Vec2::new(0.0, -0.5), Vec3::new(1.0, 0.0, 0.0)),
    Vertex::new(Vec2::new(0.5, 0.5), Vec3::new(0.0, 1.0, 0.0)),
    Vertex::new(Vec2::new(-0.5, 0.5), Vec3::new(0.0, 0.0, 1.0)),
];

const SIMPLE_SHADER: &[u32] = spirv::inline!(
    r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.position = vec4<f32>(model.position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#
);
