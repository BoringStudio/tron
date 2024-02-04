#ifndef UNIFORMS_GLOBALS_GLSL
#define UNIFORMS_GLOBALS_GLSL

#include "../math/frustum.glsl"

#define GLOBALS_SET 0
#define GLOBALS_BINDING 0

layout (set = GLOBALS_SET, binding = GLOBALS_BINDING, std140) uniform GlobalUniform {
    Frustum frustum;
    mat4 camera_view;
    mat4 camera_projection;
    mat4 camera_view_inverse;
    mat4 camera_projection_inverse;
    mat4 camera_previous_view;
    mat4 camera_previous_projection;
    uvec2 render_resolution;
    float time;
    float delta_time;
    uint frame_index;
}
globals;

#define CAMERA_VIEW globals.camera_view
#define CAMERA_PROJECTION globals.camera_projection
#define CAMERA_VIEW_INVERSE globals.camera_view_inverse
#define CAMERA_PROJECTION_INVERSE globals.camera_projection_inverse
#define CAMERA_PREVIOUS_VIEW globals.camera_previous_view
#define CAMERA_PREVIOUS_PROJECTION globals.camera_previous_projection
#define RENDER_RESOLUTION globals.render_resolution
#define TIME globals.time
#define DELTA_TIME globals.delta_time
#define FRAME_INDEX globals.frame_index

#endif  // UNIFORMS_GLOBALS_GLSL
