use vulkanalia::vk;

pub(crate) fn compute_supported_access(stages: vk::PipelineStageFlags) -> vk::AccessFlags {
    let mut res = vk::AccessFlags::empty();
    let mut bits = u32::MAX;

    while bits != 0 {
        let bit = 1 << bits.trailing_zeros();
        bits &= !bit;

        if let Some(flag) = vk::AccessFlags::from_bits(bit) {
            if access_to_stage(flag).intersects(stages) {
                res |= flag;
            }
        }
    }

    res
}

fn access_to_stage(access: vk::AccessFlags) -> vk::PipelineStageFlags {
    type AF = vk::AccessFlags;
    type PS = vk::PipelineStageFlags;

    match access {
        AF::INDIRECT_COMMAND_READ => PS::DRAW_INDIRECT,
        AF::INDEX_READ => PS::VERTEX_INPUT,
        AF::VERTEX_ATTRIBUTE_READ => PS::VERTEX_INPUT,
        AF::UNIFORM_READ => {
            PS::TASK_SHADER_EXT
                | PS::MESH_SHADER_EXT
                | PS::RAY_TRACING_SHADER_KHR
                | PS::VERTEX_SHADER
                | PS::TESSELLATION_CONTROL_SHADER
                | PS::TESSELLATION_EVALUATION_SHADER
                | PS::GEOMETRY_SHADER
                | PS::FRAGMENT_SHADER
                | PS::COMPUTE_SHADER
        }
        AF::SHADER_READ | AF::SHADER_WRITE => {
            PS::TASK_SHADER_EXT
                | PS::MESH_SHADER_EXT
                | PS::RAY_TRACING_SHADER_KHR
                | PS::VERTEX_SHADER
                | PS::TESSELLATION_CONTROL_SHADER
                | PS::TESSELLATION_EVALUATION_SHADER
                | PS::GEOMETRY_SHADER
                | PS::FRAGMENT_SHADER
                | PS::COMPUTE_SHADER
        }
        AF::INPUT_ATTACHMENT_READ => PS::FRAGMENT_SHADER,
        AF::COLOR_ATTACHMENT_READ | AF::COLOR_ATTACHMENT_WRITE => PS::COLOR_ATTACHMENT_OUTPUT,
        AF::DEPTH_STENCIL_ATTACHMENT_READ | AF::DEPTH_STENCIL_ATTACHMENT_WRITE => {
            PS::EARLY_FRAGMENT_TESTS | PS::LATE_FRAGMENT_TESTS
        }
        AF::TRANSFER_READ | AF::TRANSFER_WRITE => PS::TRANSFER,
        AF::HOST_READ | AF::HOST_WRITE => PS::HOST,
        AF::MEMORY_READ | AF::MEMORY_WRITE => PS::from_bits_truncate(!0),
        AF::ACCELERATION_STRUCTURE_READ_KHR | AF::ACCELERATION_STRUCTURE_WRITE_KHR => {
            PS::ACCELERATION_STRUCTURE_BUILD_KHR
        }
        _ => PS::empty(),
    }
}
