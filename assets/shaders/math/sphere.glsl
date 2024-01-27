#ifndef MATH_SPHERE_GLSL
#define MATH_SPHERE_GLSL

struct Sphere {
    vec4 data;
};

Sphere sphere_transform_by_mat4(Sphere sphere, mat4 transform) {
    float length0 = length(transform[0].xyz);
    float length1 = length(transform[1].xyz);
    float length2 = length(transform[2].xyz);
    float max_scale = max(max(length0, length1), length2);
    float radius = sphere.data.w * max_scale;

    return Sphere(
        vec4(
            (transform * vec4(sphere.data.xyz, 1.0)).xyz,
            sphere.data.w * max_scale
        )
    );
}

#endif  // MATH_SPHERE_GLSL
