#version 450

#extension GL_EXT_nonuniform_qualifier: require
#extension GL_ARB_shader_draw_parameters: require

#define VERTEX_POSITION 0
#define VERTEX_NORMAL 1
#define VERTEX_TANGENT 2
#define VERTEX_UV0 3
#define VERTEX_COLOR 4
#define VERTEX_ATTR_COUNT 5

#include "uniforms/globals.glsl"
#include "uniforms/bindless.glsl"
#include "uniforms/object.glsl"

layout (push_constant) uniform PushConstant {
    uint mesh_buffer_index;
    uint object_buffer_index;
    uint material_buffer_index;
} push_constant;

struct MaterialData {
    vec3 color;
};

BINDLESS_SBO_RO(std430, MaterialData, u_material_buffer);

MaterialData material_data_read(uint buffer_index, uint slot) {
    return u_material_buffer[buffer_index].items[slot];
}

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec3 out_normal;

void main() {
    ObjectData object_data = object_data_read(push_constant.object_buffer_index);
    MaterialData material_data = material_data_read(push_constant.material_buffer_index, object_data.data.z);

    Vertex vertex = vertex_read(push_constant.mesh_buffer_index, object_data.offsets);

    gl_Position = CAMERA_PROJECTION * CAMERA_VIEW * object_data.transform * vec4(vertex.position, 1.0f);
    out_color = material_data.color;
    out_normal = (object_data.transform_inverse_transpose * vec4(vertex.normal, 1.0)).xyz;
}
