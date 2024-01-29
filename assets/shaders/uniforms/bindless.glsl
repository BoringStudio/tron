#ifndef UNIFORMS_BINDLESS_GLSL
#define UNIFORMS_BINDLESS_GLSL

#include "../math/sphere.glsl"

layout (set = 1, binding = 0) uniform sampler2D global_textures[];
layout (set = 1, binding = 0) uniform usampler2D global_textures_uint[];
layout (set = 1, binding = 0) uniform sampler3D global_textures_3d[];
layout (set = 1, binding = 0) uniform usampler3D global_textures_3d_uint[];

#endif  // UNIFORMS_BINDLESS_GLSL
