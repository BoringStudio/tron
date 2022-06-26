struct CameraUniform {
    view_proj: mat4x4<f32>;
};

struct InstanceUniform {
    transform: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> camera: CameraUniform;
[[group(1), binding(0)]]
var<uniform> instance: InstanceUniform;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var v: VertexOutput;
    v.clip_position = camera.view_proj * instance.transform * vec4<f32>(vertex.position, 1.0);
    v.tex_coords = vertex.tex_coords;
    return v;
}

[[group(1), binding(1)]]
var diffuse_texture: texture_2d<f32>;
[[group(1), binding(2)]]
var diffuse_sampler: sampler;

[[stage(fragment)]]
fn fs_main(v: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(diffuse_texture, diffuse_sampler, v.tex_coords);
}
