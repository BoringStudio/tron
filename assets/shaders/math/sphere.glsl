#ifndef MATH_SPHERE_GLSL
#define MATH_SPHERE_GLSL

struct Sphere {
    vec3 center;
    float radius;
};

Sphere sphere_transform_by_mat4(Sphere sphere, mat4 transform) {
    float length0 = length(transform[0].xyz);
    float length1 = length(transform[1].xyz);
    float length2 = length(transform[2].xyz);
    float max_scale = max(max(length0, length1), length2);
    vec3 center = (transform * vec4(sphere.center, 1.0)).xyz;
    float radius = sphere.radius * max_scale;

    return Sphere(center, radius);
}

#endif  // MATH_SPHERE_GLSL
