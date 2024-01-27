#ifndef UNIFORMS_BINDLESS_GLSL
#define UNIFORMS_BINDLESS_GLSL

#include "../math/sphere.glsl"

layout (set = 1, binding = 0) uniform sampler2D global_textures[];
layout (set = 1, binding = 0) uniform usampler2D global_textures_uint[];
layout (set = 1, binding = 0) uniform sampler3D global_textures_3d[];
layout (set = 1, binding = 0) uniform usampler3D global_textures_3d_uint[];

layout (set = 1, binding = 2, std430) buffer ObjectData {
    mat4 transform;
    Sphere bounding_sphere;
    uvec4 data;
    #ifdef VERTEX_ATTR_COUNT
    float[VERTEX_ATTR_COUNT] offsets;
    #endif
};

layout (set = 1, binding = 2, std430) buffer VertexBuffer {
    float[] vertex_buffer;
};

#endif  // UNIFORMS_BINDLESS_GLSL
