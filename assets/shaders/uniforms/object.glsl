#ifndef UNIFORMS_OBJECT_GLSL
#define UNIFORMS_OBJECT_GLSL

#include "../math/sphere.glsl"
#include "./bindless.glsl"

struct ObjectData {
    mat4 transform;
    mat4 transform_inverse_transpose;
    Sphere bounding_sphere;
    uvec4 data;
    #ifdef VERTEX_ATTR_COUNT
    uint offsets[VERTEX_ATTR_COUNT];
    #endif
};

BINDLESS_SBO_RO(std430, ObjectData, u_object_data);

ObjectData object_data_read(uint buffer_index) {
    return u_object_data[buffer_index].items[gl_InstanceIndex];
}

BINDLESS_SBO_RO(std430, float, u_vertex_buffer_float);

#ifdef VERTEX_ATTR_COUNT
struct Vertex {
    #ifdef VERTEX_POSITION
    vec3 position;
    #endif

    #ifdef VERTEX_NORMAL
    vec3 normal;
    #endif

    #ifdef VERTEX_TANGENT
    vec3 tangent;
    #endif

    #ifdef VERTEX_UV0
    vec2 uv0;
    #endif

    #ifdef VERTEX_COLOR
    vec4 color;
    #endif
};

vec4 vertex_data_read_vec4(uint buffer_index, uint byte_offset) {
    uint offset = byte_offset / 4 + gl_VertexIndex * 4;
    return vec4(
        u_vertex_buffer_float[buffer_index].items[offset],
        u_vertex_buffer_float[buffer_index].items[offset + 1],
        u_vertex_buffer_float[buffer_index].items[offset + 2],
        u_vertex_buffer_float[buffer_index].items[offset + 3]
    );
}

vec3 vertex_data_read_vec3(uint buffer_index, uint byte_offset) {
    uint offset = byte_offset / 4 + gl_VertexIndex * 3;
    return vec3(
        u_vertex_buffer_float[buffer_index].items[offset],
        u_vertex_buffer_float[buffer_index].items[offset + 1],
        u_vertex_buffer_float[buffer_index].items[offset + 2]
    );
}

vec2 vertex_data_read_vec2(uint buffer_index, uint byte_offset) {
    uint offset = byte_offset / 4 + gl_VertexIndex * 2;
    return vec2(
        u_vertex_buffer_float[buffer_index].items[offset],
        u_vertex_buffer_float[buffer_index].items[offset + 1]
    );
}

Vertex vertex_read(uint buffer_index, uint[VERTEX_ATTR_COUNT] offsets) {
    Vertex result;

    #ifdef VERTEX_POSITION
    result.position = vertex_data_read_vec3(buffer_index, offsets[VERTEX_POSITION]);
    #endif
    #ifdef VERTEX_NORMAL
    result.normal = vertex_data_read_vec3(buffer_index, offsets[VERTEX_NORMAL]);
    #endif
    #ifdef VERTEX_TANGENT
    result.tangent = vertex_data_read_vec3(buffer_index, offsets[VERTEX_TANGENT]);
    #endif
    #ifdef VERTEX_UV0
    result.uv0 = vertex_data_read_vec2(buffer_index, offsets[VERTEX_UV0]);
    #endif
    #ifdef VERTEX_COLOR
    result.color = vertex_data_read_vec4(buffer_index, offsets[VERTEX_COLOR]);
    #endif

    return result;
}

#endif // VERTEX_ATTR_COUNT

#endif // UNIFORMS_OBJECT_GLSL
