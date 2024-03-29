#ifndef MATH_FRUSTUM_GLSL
#define MATH_FRUSTUM_GLSL

#include "./sphere.glsl"

struct Plane {
    vec4 inner;
};

float plane_distance_to_point(Plane plane, vec3 point) {
    return dot(plane.inner.xyz, point) + plane.inner.w;
}

struct Frustum {
    Plane left;
    Plane right;
    Plane top;
    Plane bottom;
    Plane near;
};

bool frustum_contains_sphere(Frustum frustum, Sphere sphere) {
    float neg_radius = -sphere.data.w;
    return plane_distance_to_point(frustum.left, sphere.data.xyz) >= neg_radius &&
    plane_distance_to_point(frustum.right, sphere.data.xyz) >= neg_radius &&
    plane_distance_to_point(frustum.top, sphere.data.xyz) >= neg_radius &&
    plane_distance_to_point(frustum.bottom, sphere.data.xyz) >= neg_radius &&
    plane_distance_to_point(frustum.near, sphere.data.xyz) >= neg_radius;
}

#endif  // MATH_FRUSTUM_GLSL
