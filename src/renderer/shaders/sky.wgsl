struct CameraUniform {
    view_proj: mat4x4<f32>,
}

struct SunUniform {
    rayleigh: f32,
    turbidity: f32,
    mie_coefficient: f32,
    luminance: f32,
    // direction (xyz), mieDirectionalG (w)
    direction: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(1) @binding(0)
var<uniform> sun: SunUniform;

const up = vec3<f32>(0.0, 1.0, 0.0);
const camera_position = vec3<f32>(0.0, 0.0, 0.0);

const sun_angular_diameter_cos = 0.999812627955556656903750820965829532378703448007194460804;

const pi = 3.141592653589793238462643383279502884197169;

// lambda = vec3(680e-9, 550e-9, 450e-9)
// (8.0 * pow(pi, 3.0) * pow(pow(n, 2.0) - 1.0, 2.0) * (6.0 + 3.0 * pn)) / (3.0 * N * pow(lambda, vec3(4.0)) * (6.0 - 7.0 * pn))
const total_rayleigh = vec3<f32>(5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5);

// v = 4
// K = vec3(0.686, 0.678, 0.666);
// MIE = pi * pow( ( 2.0 * pi ) / lambda, vec3( v - 2.0 ) ) * K
const mie = vec3<f32>(1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14);

// earth shadow hack
fn sun_intensity(zenith_angle_cos: f32) -> f32 {
    // pi / 1.95;
    let cutoff_angle: f32 = 1.6110731556870734;
    let steepness: f32 = 1.5;
    let ee: f32 = 1000.0;

    let clamped_zenith_angle_cos = clamp(zenith_angle_cos, -1.0, 1.0);
    return ee * max(0.0, 1.0 - exp(-(cutoff_angle - acos(clamped_zenith_angle_cos)) / steepness));
}

fn total_mie(t: f32) -> vec3<f32> {
    return t * 8.68e-19 * mie;
}

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(3) sun_intensity: f32,
    @location(4) sun_fade: f32,
    @location(1) beta_r: vec3<f32>,
    @location(2) beta_m: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    let view_proj = mat4x4<f32>(
        camera.view_proj[0],
        camera.view_proj[1],
        camera.view_proj[2],
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    var v: VertexOutput;
    v.clip_position = view_proj * vec4<f32>(vertex.position, 0.0);
    v.clip_position.z = v.clip_position.w;
    v.position = vertex.position;
    v.sun_intensity = sun_intensity(dot(sun.direction.xyz, up));
    v.sun_fade = 1.0 - clamp(1.0 - exp(sun.direction.y / 450000.0), 0.0, 1.0);

    let c = sun.rayleigh + v.sun_fade - 1.0;
    v.beta_r = total_rayleigh * c;
    v.beta_m = total_mie(sun.turbidity) * sun.mie_coefficient;

    return v;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    let three_over_sixteen_pi = 0.05968310365946075;
    return three_over_sixteen_pi * (1.0 + pow(cos_theta, 2.0));
}

fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let one_over_four_pi = 0.07957747154594767;
    let g2 = pow(g, 2.0);
    let inverse = 1.0 / pow(1.0 - 2.0 * g * cos_theta + g2, 1.5);
    return one_over_four_pi * (1.0 - g2) * inverse;
}

const white_scale = 1.0748724675633854; // 1.0 / Uncharted2Tonemap(1000.0)

fn uncharted_tonemap(x: vec3<f32>) -> vec3<f32> {
    let A: f32 = 0.15;
    let B: f32 = 0.50;
    let C: f32 = 0.10;
    let D: f32 = 0.20;
    let E: f32 = 0.02;
    let F: f32 = 0.30;

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

@fragment
fn fs_main(v: VertexOutput) -> @location(0) vec4<f32> {
    let rayleigh_zenith_length = 8.4e3;
    let mie_zenith_length = 1.25e3;

    // optical length
    let zenith_angle = acos(max(0.0, dot(up, normalize(v.position - camera_position))));
    let inverse = 1.0 / (cos(zenith_angle) + 0.15 * pow(93.885 - ((zenith_angle * 180.0) / pi), -1.253));
    let sR = rayleigh_zenith_length * inverse;
    let sM = mie_zenith_length * inverse;

    // combined extinction factor
    let fex = exp(-(v.beta_r * sR + v.beta_m * sM));

    // in scattering
    let cos_theta = dot(normalize(v.position - camera_position), sun.direction.xyz);

    let r_phase = rayleigh_phase(cos_theta * 0.5 + 0.5);
    let beta_r_theta = v.beta_r * r_phase;

    let m_phase = hg_phase(cos_theta, sun.direction.w);
    let beta_m_theta = v.beta_m * m_phase;

    var lin = pow(v.sun_intensity * ((beta_r_theta + beta_m_theta) / (v.beta_r + v.beta_m)) * (1.0 - fex), vec3<f32>(1.5));
    lin *= mix(vec3<f32>(1.0), pow(v.sun_intensity * ((beta_r_theta + beta_m_theta) / (v.beta_r + v.beta_m)) * fex, vec3<f32>(1.0 / 2.0)), clamp(pow(1.0 - dot(up, sun.direction.xyz), 5.0), 0.0, 1.0));

    // nightsky
    let direction = normalize(v.position - camera_position);
    let theta = acos(direction.y); // elevation --> y-axis, [-pi/2, pi/2]
    let phi = atan2(direction.z, direction.x); // azimuth --> x-axis [-pi/2, pi/2]
    let uv = vec2(phi, theta) / vec2(2.0 * pi, pi) + vec2(0.5, 0.0);
    var l0 = vec3<f32>(0.1) * fex;

    // composition + solar disc
    let sundisk = smoothstep(sun_angular_diameter_cos, sun_angular_diameter_cos + 0.00002, cos_theta);
    l0 += v.sun_intensity * 19000.0 * fex * sundisk;

    let tex_color = (lin + l0) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075);

    let curr = uncharted_tonemap((log2(2.0 / pow(sun.luminance, 4.0))) * tex_color);
    let color = curr * white_scale;

    let ret_color = pow(color, vec3<f32>(1.0 / (1.2 + (1.2 * v.sun_fade))));

    return vec4<f32>(ret_color, 1.0);
}
