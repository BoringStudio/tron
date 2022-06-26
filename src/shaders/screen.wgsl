struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[builtin(vertex_index)]] index: u32
) -> VertexOutput {
    var v: VertexOutput;
    let low_bit = i32(index & 0x1u);
    let high_bit = i32(index >> 1u);

    v.position = vec4<f32>(
        f32(4 * low_bit - 1),
        f32(4 * high_bit - 1),
        0.0,
        1.0
    );

    v.tex_coords = vec2<f32>(
        f32(2 * low_bit),
        f32(1 - 2 * high_bit)
    );

    return v;
}

[[group(0), binding(0)]]
var screen_texture: texture_2d<f32>;
[[group(0), binding(1)]]
var screen_sampler: sampler;

[[stage(fragment)]]
fn fs_main(
    v: VertexOutput,
) -> [[location(0)]] vec4<f32> {
    return textureSample(screen_texture, screen_sampler, v.tex_coords);
}
