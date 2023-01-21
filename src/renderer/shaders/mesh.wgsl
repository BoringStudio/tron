struct BaseUniform {
    view_proj: mat4x4<f32>,
    view_width: u32,
    view_height: u32,
    time: f32,
}

struct InstanceUniform {
    transform: mat3x4<f32>,
    normal_matrix: mat3x4<f32>,
}

@group(0) @binding(0)
var<uniform> base: BaseUniform;
@group(1) @binding(0)
var<uniform> instance: InstanceUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var v: VertexOutput;
    v.clip_position = base.view_proj * mat4x4(
        instance.transform[0],
        instance.transform[1],
        instance.transform[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    ) * vec4<f32>(vertex.position, 1.0);
    v.normal = mat3x3(instance.normal_matrix[0].xyz, instance.normal_matrix[1].xyz, instance.normal_matrix[2].xyz) * vertex.normal;
    v.tex_coords = vertex.tex_coords;
    return v;
}

@group(1) @binding(1)
var diffuse_texture: texture_2d<f32>;
@group(1) @binding(2)
var diffuse_sampler: sampler;

@fragment
fn fs_main(v: VertexOutput) -> @location(0) vec4<f32> {
    var shift = vec2<f32>((base.time * 0.1), 0.0);

    var direct_light = max(dot(v.normal, vec3<f32>(0.1, -1.0, 0.1)), 0.0);
    var color = textureSample(diffuse_texture, diffuse_sampler, v.tex_coords + shift).xyz;
    color *= min(0.5 + direct_light, 1.0);
    return vec4(color, 1.0);
}
