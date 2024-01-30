#ifndef UNIFORMS_BINDLESS_GLSL
#define UNIFORMS_BINDLESS_GLSL

#include "../math/sphere.glsl"

#define BINDLESS_SET 1
#define BINDLESS_TEX_BINDING 0
#define BINDLESS_UBO_BINDING 1
#define BINDLESS_SBO_BINDING 2

#define BINDLESS_TEX_COUNT 1024
#define BINDLESS_UBO_COUNT 1024
#define BINDLESS_SBO_COUNT 1024

layout (set = BINDLESS_SET, binding = BINDLESS_TEX_BINDING) uniform sampler2D u_global_textures[];
layout (set = BINDLESS_SET, binding = BINDLESS_TEX_BINDING) uniform usampler2D u_global_textures_uint[];
layout (set = BINDLESS_SET, binding = BINDLESS_TEX_BINDING) uniform sampler3D u_global_textures_3d[];
layout (set = BINDLESS_SET, binding = BINDLESS_TEX_BINDING) uniform usampler3D u_global_textures_3d_uint[];

#endif  // UNIFORMS_BINDLESS_GLSL
