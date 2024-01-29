#version 450

#extension GL_EXT_nonuniform_qualifier: require
#extension GL_ARB_shader_draw_parameters: require

#include "uniforms/bindless.glsl"
#include "uniforms/globals.glsl"

struct ObjectData {
    mat4 transform;
    mat4 transform_inverse_transpose;
    Sphere bounding_sphere;
    uvec4 data;
    uint offsets[5];
};

layout (set = 1, binding = 2, std430) buffer ObjectDataBuffer {
    ObjectData items[];
} u_object_data[1024];

layout (set = 1, binding = 2, std430) buffer FloatVertexBuffer {
    float data[];
} u_vertex_buffer_float[1024];

layout (set = 1, binding = 2, std430) buffer UintVertexBuffer {
    uint data[];
} u_vertex_buffer_uint[1024];

layout (set = 1, binding = 2, std430) buffer MaterialBuffer {
    vec3 colors[];
} u_material_buffer[1024];

layout (push_constant) uniform PushConstant {
    uint mesh_buffer_index;
    uint object_buffer_index;
    uint material_buffer_index;
} push_constant;

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec3 out_normal;

vec3 vertex_data_read_vec3(uint buffer_index, uint byte_offset) {
    uint offset = byte_offset / 4 + gl_VertexIndex * 3;
    return vec3(
        u_vertex_buffer_float[buffer_index].data[offset],
        u_vertex_buffer_float[buffer_index].data[offset + 1],
        u_vertex_buffer_float[buffer_index].data[offset + 2]
    );
}

vec2 vertex_data_read_vec2(uint buffer_index, uint byte_offset) {
    uint offset = byte_offset / 4 + gl_VertexIndex * 2;
    return vec2(
        u_vertex_buffer_float[buffer_index].data[offset],
        u_vertex_buffer_float[buffer_index].data[offset + 1]
    );
}

uvec3 vertex_data_read_uvec3(uint buffer_index, uint byte_offset) {
    uint offset = byte_offset / 4 + gl_VertexIndex * 3;
    return uvec3(
        u_vertex_buffer_uint[buffer_index].data[offset],
        u_vertex_buffer_uint[buffer_index].data[offset + 1],
        u_vertex_buffer_uint[buffer_index].data[offset + 2]
    );
}

void main() {
    ObjectData object_data = u_object_data[push_constant.object_buffer_index].items[gl_InstanceIndex];
    vec3 color = u_material_buffer[push_constant.material_buffer_index].colors[object_data.data.z];

    vec3 position = vertex_data_read_vec3(push_constant.mesh_buffer_index, object_data.offsets[0]);
    vec3 normal = vertex_data_read_vec3(push_constant.mesh_buffer_index, object_data.offsets[1]);

    gl_Position = CAMERA_PROJECTION * CAMERA_VIEW * object_data.transform * vec4(position, 1.0f);
    out_color = color;
    out_normal = (object_data.transform_inverse_transpose * vec4(normal, 1.0)).xyz;
}
