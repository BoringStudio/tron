#version 450

#include <uniforms/bindless.glsl>
#include <uniforms/globals.glsl>

layout(location = 0) out vec3 outColor;

void main() {
  const vec3 positions[3] = vec3[3](vec3(0.5f, 0.5f, 0.0f), vec3(-0.5f, 0.5f, 0.0f), vec3(0.0f, -0.5f, 0.0f));
  const vec3 colors[3] = vec3[3](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));

  vec3 prev_color = colors[gl_VertexIndex];
  vec3 next_color = colors[(gl_VertexIndex + 1) % 3];

  float image_aspect = RENDER_RESOLUTION.x / RENDER_RESOLUTION.y;
  vec3 position = positions[gl_VertexIndex];
  position.x /= image_aspect;

  gl_Position = vec4(position, 1.0f);
  outColor = mix(prev_color, next_color, sin(TIME));
}
