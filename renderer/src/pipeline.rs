use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::base::RendererBase;
use super::pipeline_layout::PipelineLayout;
use super::shader_module::ShaderModule;

pub struct Pipeline {
    base: Rc<RendererBase>,
    pipeline_layout: PipelineLayout,
    render_pass: SimpleRenderPass,
    handle: vk::Pipeline,
}

impl Pipeline {
    pub unsafe fn new(base: Rc<RendererBase>) -> Result<Self> {
        let mesh_shader = ShaderModule::new(base.clone(), TRIANGLE_SHADER)?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(mesh_shader.handle())
            .name(b"vs_main\0");
        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(mesh_shader.handle())
            .name(b"fs_main\0");

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

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

        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

        // TODO: use builders
        let pipeline_layout = PipelineLayout::new(base.clone())?;
        let render_pass = SimpleRenderPass::new(base.clone())?;

        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
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
    unsafe fn new(base: Rc<RendererBase>) -> Result<Self> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(
                base.physical_device()
                    .swapchain_support
                    .surface_format
                    .format,
            )
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

const TRIANGLE_SHADER: &[u32] = spirv::inline!(
    r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    var v: VertexOutput;
    let low_bit = i32(index & 0x1u);
    let high_bit = i32(index >> 1u);

    v.position = vec4<f32>(
        f32(4 * low_bit - 1),
        f32(4 * high_bit - 1),
        0.0,
        1.0
    );

    v.tex_coords = vec2<f32>(
        f32(2 * low_bit),
        f32(1 - 2 * high_bit)
    );

    return v;
}

@fragment
fn fs_main(v: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#
);
