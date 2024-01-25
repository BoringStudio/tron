#ifndef MATH_COLOR_GLSL
#define MATH_COLOR_GLSL

float saturate(float v) {
  return clamp(v, 0.0, 1.0);
}

#endif  // MATH_COLOR_GLSL
