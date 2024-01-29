#version 450

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec3 in_normal;

layout (location = 0) out vec4 out_frag_color;

void main() {
    const vec3 light_direction = normalize(vec3(0.5, 0.5, 0.5));

    vec3 color = clamp(dot(light_direction, in_normal), 0.0, 1.0) * in_color;

    out_frag_color = vec4(color, 1.0f);
}
