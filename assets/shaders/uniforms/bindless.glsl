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

#define BINDLESS_TEX(ty, name) \
layout (set = BINDLESS_SET, binding = BINDLESS_TEX_BINDING) uniform ty name[BINDLESS_TEX_COUNT]

BINDLESS_TEX(sampler2D, u_global_textures);
BINDLESS_TEX(usampler2D, u_global_textures_uint);
BINDLESS_TEX(sampler3D, u_global_textures_3d);
BINDLESS_TEX(usampler3D, u_global_textures_3d_uint);

#define BINDLESS_UBO(ty, name) \
layout (set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform ty##Buffer { \
ty items[]; \
} name[BINDLESS_UBO_COUNT]

#define BINDLESS_SBO_RO(layout_, ty_, name_) \
layout (set = BINDLESS_SET, binding = BINDLESS_SBO_BINDING, layout_) readonly buffer ty_##Buffer { \
ty_ items[]; \
} name_[BINDLESS_SBO_COUNT]

struct DummyUniform { uint ignore; };
BINDLESS_UBO(DummyUniform, u_dummy_ubo);
BINDLESS_SBO_RO(std430, DummyUniform, u_dummy_sbo);

#endif  // UNIFORMS_BINDLESS_GLSL
