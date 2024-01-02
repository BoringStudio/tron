use std::sync::Arc;

use glam::{IVec2, UVec2, UVec3};
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{
    CompareOp, ComputeShader, FragmentShader, PipelineLayout, RenderPass, VertexShader,
};
use crate::types::State;
use crate::util::{FromGfx, ToVk};

/// A three-dimensional subregion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub x: Bounds,
    pub y: Bounds,
    pub z: Bounds,
}

impl From<Rect> for Viewport {
    fn from(value: Rect) -> Self {
        Self {
            x: Bounds::new(value.offset.x as f32, value.extent.x as f32),
            y: Bounds::new(value.offset.y as f32, value.extent.y as f32),
            z: Bounds::new(0.0, 1.0),
        }
    }
}

impl FromGfx<Viewport> for vk::Viewport {
    fn from_gfx(value: Viewport) -> Self {
        Self {
            x: value.x.offset,
            y: value.y.offset,
            width: value.x.size,
            height: value.y.size,
            min_depth: value.z.offset,
            max_depth: value.z.offset + value.z.size,
        }
    }
}

impl From<UVec2> for Viewport {
    fn from(value: UVec2) -> Self {
        Self {
            x: Bounds::new(0.0, value.x as f32),
            y: Bounds::new(0.0, value.y as f32),
            z: Bounds::new(0.0, 1.0),
        }
    }
}

impl From<UVec3> for Viewport {
    fn from(value: UVec3) -> Self {
        Self {
            x: Bounds::new(0.0, value.x as f32),
            y: Bounds::new(0.0, value.y as f32),
            z: Bounds::new(0.0, value.z as f32),
        }
    }
}

impl From<UVec2> for State<Viewport> {
    fn from(value: UVec2) -> Self {
        Self::Static(value.into())
    }
}

impl From<UVec3> for State<Viewport> {
    fn from(value: UVec3) -> Self {
        Self::Static(value.into())
    }
}

/// A two-dimensional subregion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub offset: IVec2,
    pub extent: UVec2,
}

impl FromGfx<Rect> for vk::Rect2D {
    fn from_gfx(value: Rect) -> Self {
        Self {
            offset: value.offset.to_vk(),
            extent: value.extent.to_vk(),
        }
    }
}

impl From<UVec2> for Rect {
    fn from(extent: UVec2) -> Self {
        Self {
            offset: IVec2::ZERO,
            extent,
        }
    }
}

impl From<UVec2> for State<Rect> {
    fn from(value: UVec2) -> Self {
        Self::Static(value.into())
    }
}

/// A one-dimensional subregion.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub offset: f32,
    pub size: f32,
}

impl Bounds {
    pub fn new(offset: f32, size: f32) -> Self {
        Self { offset, size }
    }
}

impl Eq for Bounds {}
impl PartialEq for Bounds {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        f32::to_bits(self.offset) == f32::to_bits(other.offset)
            && f32::to_bits(self.size) == f32::to_bits(other.size)
    }
}

impl std::hash::Hash for Bounds {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(f32::to_bits(self.offset));
        state.write_u32(f32::to_bits(self.size));
    }
}

/// Specify the bind point of a pipeline object to a command buffer.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PipelineBindPoint {
    Graphics,
    Compute,
}

impl FromGfx<PipelineBindPoint> for vk::PipelineBindPoint {
    fn from_gfx(value: PipelineBindPoint) -> Self {
        match value {
            PipelineBindPoint::Graphics => Self::GRAPHICS,
            PipelineBindPoint::Compute => Self::COMPUTE,
        }
    }
}

// === Graphics pipeline ===

/// Structure specifying parameters of a newly created graphics pipeline.
#[derive(Debug, Clone)]
pub struct GraphicsPipelineInfo {
    pub descr: GraphicsPipelineDescr,
    pub rendering: GraphicsPipelineRenderingInfo,
}

/// Graphics pipeline structure description.
#[derive(Debug, Clone)]
pub struct GraphicsPipelineDescr {
    pub vertex_bindings: Vec<VertexInputBinding>,
    pub vertex_attributes: Vec<VertexInputAttribute>,
    pub primitive_topology: PrimitiveTopology,
    pub primitive_restart_enable: bool,
    pub vertex_shader: VertexShader,
    pub rasterizer: Option<Rasterizer>,
    pub layout: PipelineLayout,
}

/// Graphics pipeline rasterization stage parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Rasterizer {
    pub viewport: State<vk::Viewport>,
    pub scissor: State<vk::Rect2D>,
    pub depth_clamp: bool,
    pub front_face: FrontFace,
    pub cull_mode: Option<CullMode>,
    pub polygin_mode: PolygonMode,
    pub depth_test: Option<DepthTest>,
    pub stencil_tests: Option<StencilTests>,
    pub depth_bounds: Option<State<Bounds>>,
    pub fragment_shader: Option<FragmentShader>,
    pub color_blend: ColorBlend,
}

impl Default for Rasterizer {
    fn default() -> Self {
        Self {
            viewport: State::Dynamic,
            scissor: State::Dynamic,
            depth_clamp: false,
            front_face: FrontFace::CW,
            cull_mode: None,
            polygin_mode: PolygonMode::Fill,
            depth_test: None,
            stencil_tests: None,
            depth_bounds: None,
            fragment_shader: None,
            color_blend: ColorBlend::default(),
        }
    }
}

/// Graphics pipeline rendering stage parameters.
#[derive(Debug, Clone)]
pub struct GraphicsPipelineRenderingInfo {
    pub render_pass: RenderPass,
    pub subpass: u32,
}

/// Graphics pipeline vertex binding parameters.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VertexInputBinding {
    pub rate: VertexInputRate,
    pub stride: u32,
}

/// Graphics pipeline vertex attribute parameters.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VertexInputAttribute {
    /// The shader input location number for this attribute.
    pub location: u32,
    /// The binding number which this attribute takes its data from.
    pub binding: u32,
    /// The size and type of the vertex attribute data.
    pub format: VertexFormat,
    /// A byte offset of this attribute relative to the start of an element
    /// in the vertex input binding.
    pub offset: u32,
}

impl FromGfx<VertexInputAttribute> for vk::VertexInputAttributeDescription {
    fn from_gfx(value: VertexInputAttribute) -> Self {
        Self {
            location: value.location,
            binding: value.binding,
            format: value.format.to_vk(),
            offset: value.offset,
        }
    }
}

/// Vertex Format for a [`VertexInputAttribute`] (input).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum VertexFormat {
    /// Two unsigned bytes (u8). `vec2<u32>` in shaders.
    Uint8x2 = 0,
    /// Four unsigned bytes (u8). `vec4<u32>` in shaders.
    Uint8x4 = 1,
    /// Two signed bytes (i8). `vec2<i32>` in shaders.
    Sint8x2 = 2,
    /// Four signed bytes (i8). `vec4<i32>` in shaders.
    Sint8x4 = 3,
    /// Two unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec2<f32>` in shaders.
    Unorm8x2 = 4,
    /// Four unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec4<f32>` in shaders.
    Unorm8x4 = 5,
    /// Two signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec2<f32>` in shaders.
    Snorm8x2 = 6,
    /// Four signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec4<f32>` in shaders.
    Snorm8x4 = 7,
    /// Two unsigned shorts (u16). `vec2<u32>` in shaders.
    Uint16x2 = 8,
    /// Four unsigned shorts (u16). `vec4<u32>` in shaders.
    Uint16x4 = 9,
    /// Two signed shorts (i16). `vec2<i32>` in shaders.
    Sint16x2 = 10,
    /// Four signed shorts (i16). `vec4<i32>` in shaders.
    Sint16x4 = 11,
    /// Two unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec2<f32>` in shaders.
    Unorm16x2 = 12,
    /// Four unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec4<f32>` in shaders.
    Unorm16x4 = 13,
    /// Two signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec2<f32>` in shaders.
    Snorm16x2 = 14,
    /// Four signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec4<f32>` in shaders.
    Snorm16x4 = 15,
    /// Two half-precision floats (no Rust equiv). `vec2<f32>` in shaders.
    Float16x2 = 16,
    /// Four half-precision floats (no Rust equiv). `vec4<f32>` in shaders.
    Float16x4 = 17,
    /// One single-precision float (f32). `f32` in shaders.
    Float32 = 18,
    /// Two single-precision floats (f32). `vec2<f32>` in shaders.
    Float32x2 = 19,
    /// Three single-precision floats (f32). `vec3<f32>` in shaders.
    Float32x3 = 20,
    /// Four single-precision floats (f32). `vec4<f32>` in shaders.
    Float32x4 = 21,
    /// One unsigned int (u32). `u32` in shaders.
    Uint32 = 22,
    /// Two unsigned ints (u32). `vec2<u32>` in shaders.
    Uint32x2 = 23,
    /// Three unsigned ints (u32). `vec3<u32>` in shaders.
    Uint32x3 = 24,
    /// Four unsigned ints (u32). `vec4<u32>` in shaders.
    Uint32x4 = 25,
    /// One signed int (i32). `i32` in shaders.
    Sint32 = 26,
    /// Two signed ints (i32). `vec2<i32>` in shaders.
    Sint32x2 = 27,
    /// Three signed ints (i32). `vec3<i32>` in shaders.
    Sint32x3 = 28,
    /// Four signed ints (i32). `vec4<i32>` in shaders.
    Sint32x4 = 29,
    /// One double-precision float (f64). `f32` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64 = 30,
    /// Two double-precision floats (f64). `vec2<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x2 = 31,
    /// Three double-precision floats (f64). `vec3<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x3 = 32,
    /// Four double-precision floats (f64). `vec4<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x4 = 33,
}

impl VertexFormat {
    /// Returns the byte size of the format.
    pub const fn size(&self) -> u32 {
        match self {
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Uint8x4
            | Self::Sint8x4
            | Self::Unorm8x4
            | Self::Snorm8x4
            | Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Float16x2
            | Self::Float32
            | Self::Uint32
            | Self::Sint32 => 4,
            Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4
            | Self::Float16x4
            | Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2
            | Self::Float64 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 | Self::Float64x2 => 16,
            Self::Float64x3 => 24,
            Self::Float64x4 => 32,
        }
    }
}

impl FromGfx<VertexFormat> for vk::Format {
    fn from_gfx(value: VertexFormat) -> Self {
        match value {
            VertexFormat::Uint8x2 => Self::R8G8_UINT,
            VertexFormat::Uint8x4 => Self::R8G8B8A8_UINT,
            VertexFormat::Sint8x2 => Self::R8G8_SINT,
            VertexFormat::Sint8x4 => Self::R8G8B8A8_SINT,
            VertexFormat::Unorm8x2 => Self::R8G8_UNORM,
            VertexFormat::Unorm8x4 => Self::R8G8B8A8_UNORM,
            VertexFormat::Snorm8x2 => Self::R8G8_SNORM,
            VertexFormat::Snorm8x4 => Self::R8G8B8A8_SNORM,
            VertexFormat::Uint16x2 => Self::R16G16_UINT,
            VertexFormat::Uint16x4 => Self::R16G16B16A16_UINT,
            VertexFormat::Sint16x2 => Self::R16G16_SINT,
            VertexFormat::Sint16x4 => Self::R16G16B16A16_SINT,
            VertexFormat::Unorm16x2 => Self::R16G16_UNORM,
            VertexFormat::Unorm16x4 => Self::R16G16B16A16_UNORM,
            VertexFormat::Snorm16x2 => Self::R16G16_SNORM,
            VertexFormat::Snorm16x4 => Self::R16G16B16A16_SNORM,
            VertexFormat::Float16x2 => Self::R16G16_SFLOAT,
            VertexFormat::Float16x4 => Self::R16G16B16A16_SFLOAT,
            VertexFormat::Float32 => Self::R32_SFLOAT,
            VertexFormat::Float32x2 => Self::R32G32_SFLOAT,
            VertexFormat::Float32x3 => Self::R32G32B32_SFLOAT,
            VertexFormat::Float32x4 => Self::R32G32B32A32_SFLOAT,
            VertexFormat::Uint32 => Self::R32_UINT,
            VertexFormat::Uint32x2 => Self::R32G32_UINT,
            VertexFormat::Uint32x3 => Self::R32G32B32_UINT,
            VertexFormat::Uint32x4 => Self::R32G32B32A32_UINT,
            VertexFormat::Sint32 => Self::R32_SINT,
            VertexFormat::Sint32x2 => Self::R32G32_SINT,
            VertexFormat::Sint32x3 => Self::R32G32B32_SINT,
            VertexFormat::Sint32x4 => Self::R32G32B32A32_SINT,
            VertexFormat::Float64 => Self::R64_SFLOAT,
            VertexFormat::Float64x2 => Self::R64G64_SFLOAT,
            VertexFormat::Float64x3 => Self::R64G64B64_SFLOAT,
            VertexFormat::Float64x4 => Self::R64G64B64A64_SFLOAT,
        }
    }
}

/// Specify rate at which vertex attributes are pulled from buffers.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum VertexInputRate {
    /// Vertex attribute addressing is a function of the vertex index.
    Vertex,
    /// Vertex attribute addressing is a function of the instance index.
    Instance,
}

impl FromGfx<VertexInputRate> for vk::VertexInputRate {
    #[inline]
    fn from_gfx(value: VertexInputRate) -> Self {
        match value {
            VertexInputRate::Vertex => Self::VERTEX,
            VertexInputRate::Instance => Self::INSTANCE,
        }
    }
}

/// Supported primitive topologies.
#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PrimitiveTopology {
    /// Separate point primitives.
    PointList,
    /// Separate line primitives.
    LineList,
    /// A series of connected line segments with consecutive lines sharing a vertex.
    LineStrip,
    /// Separate triangle primitives.
    #[default]
    TriangleList,
    /// A series of connected triangles with consecutive triangles sharing an edge.
    TriangleStrip,
    /// A series of connected triangles sharing a central vertex.
    TriangleFan,
}

impl FromGfx<PrimitiveTopology> for vk::PrimitiveTopology {
    fn from_gfx(value: PrimitiveTopology) -> Self {
        match value {
            PrimitiveTopology::PointList => Self::POINT_LIST,
            PrimitiveTopology::LineList => Self::LINE_LIST,
            PrimitiveTopology::LineStrip => Self::LINE_STRIP,
            PrimitiveTopology::TriangleList => Self::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => Self::TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => Self::TRIANGLE_FAN,
        }
    }
}

/// Interpret polygon front-facing orientation.
#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum FrontFace {
    /// Clockwise.
    #[default]
    CW,
    /// Counter-clockwise.
    CCW,
}

impl FromGfx<FrontFace> for vk::FrontFace {
    fn from_gfx(value: FrontFace) -> Self {
        match value {
            FrontFace::CW => Self::CLOCKWISE,
            FrontFace::CCW => Self::COUNTER_CLOCKWISE,
        }
    }
}

/// Triangle culling mode.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CullMode {
    /// Cull front-facing triangles.
    Front,
    /// Cull back-facing triangles.
    Back,
    /// Cull both front- and back-facing triangles.
    FrontAndBack,
}

impl FromGfx<CullMode> for vk::CullModeFlags {
    fn from_gfx(value: CullMode) -> Self {
        match value {
            CullMode::Front => Self::FRONT,
            CullMode::Back => Self::BACK,
            CullMode::FrontAndBack => Self::FRONT_AND_BACK,
        }
    }
}

impl FromGfx<Option<CullMode>> for vk::CullModeFlags {
    fn from_gfx(value: Option<CullMode>) -> Self {
        match value {
            Some(value) => value.to_vk(),
            None => Self::NONE,
        }
    }
}

/// Polygon rasterization mode.
#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PolygonMode {
    /// Polygons are rendered as solid surfaces.
    #[default]
    Fill,
    /// Polygon edges are drawn as line segments.
    Line,
    /// Polygon vertices are drawn as points.
    Point,
}

impl FromGfx<PolygonMode> for vk::PolygonMode {
    #[inline]
    fn from_gfx(value: PolygonMode) -> Self {
        match value {
            PolygonMode::Fill => Self::FILL,
            PolygonMode::Line => Self::LINE,
            PolygonMode::Point => Self::POINT,
        }
    }
}

/// Depth test parameters.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DepthTest {
    pub compare: CompareOp,
    pub write: bool,
}

/// Stencil test parameters for both faces.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct StencilTests {
    pub front: StencilTest,
    pub back: StencilTest,
}

/// Stencil test parameters for a single face.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct StencilTest {
    pub compare: CompareOp,
    pub compare_mask: State<u32>,
    pub write_mask: State<u32>,
    pub reference: State<u32>,
    pub fail: StencilOp,
    pub pass: StencilOp,
    pub depth_fail: StencilOp,
}

/// Stencil comparison function.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StencilOp {
    /// Keeps the current value.
    Keep,
    /// Sets the value to `0`.
    Zero,
    /// Sets the value to `reference`.
    Replace,
    /// Increments the current value and clamps to the maximum representable unsigned value.
    IncrementAndClamp,
    /// Decrements the current value and clamps to `0`.
    DecrementAndClamp,
    /// Bitwise-inverts the current value.
    Invert,
    /// Increments the current value and wraps to `0` when the maximum value would have been
    /// exceeded.
    IncrementAndWrap,
    /// Decrements the current value and wraps to the maximum representable unsigned value when
    /// the value would go below `0`.
    DecrementAndWrap,
}

impl FromGfx<StencilOp> for vk::StencilOp {
    fn from_gfx(value: StencilOp) -> Self {
        match value {
            StencilOp::Keep => Self::KEEP,
            StencilOp::Zero => Self::ZERO,
            StencilOp::Replace => Self::REPLACE,
            StencilOp::IncrementAndClamp => Self::INCREMENT_AND_CLAMP,
            StencilOp::DecrementAndClamp => Self::DECREMENT_AND_CLAMP,
            StencilOp::Invert => Self::INVERT,
            StencilOp::IncrementAndWrap => Self::INCREMENT_AND_WRAP,
            StencilOp::DecrementAndWrap => Self::DECREMENT_AND_WRAP,
        }
    }
}

/// Structure specifying parameters of a newly created pipeline color blend state.
#[derive(Debug, Clone)]
pub enum ColorBlend {
    /// Color attachment blending uses a logical operation.
    Logic { op: LogicOp },
    /// Color blending is controlled by the same parameters for all attachments.
    Blending {
        blending: Option<Blending>,
        write_mask: ComponentMask,
        constants: State<[f32; 4]>,
    },
    /// Color blending is controlled per-attachment.
    IndependentBlending {
        blending: Vec<(Option<Blending>, ComponentMask)>,
        constants: State<[f32; 4]>,
    },
}

impl Eq for ColorBlend {}
impl PartialEq for ColorBlend {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Logic { op: l_op }, Self::Logic { op: r_op }) => l_op.eq(r_op),
            (
                Self::Blending {
                    blending: l_blending,
                    write_mask: l_write_mask,
                    constants: l_constants,
                },
                Self::Blending {
                    blending: r_blending,
                    write_mask: r_write_mask,
                    constants: r_constants,
                },
            ) => {
                l_blending.eq(r_blending)
                    && l_write_mask.eq(r_write_mask)
                    && eq_constants(l_constants, r_constants)
            }
            (
                Self::IndependentBlending {
                    blending: l_blending,
                    constants: l_constants,
                },
                Self::IndependentBlending {
                    blending: r_blending,
                    constants: r_constants,
                },
            ) => l_blending.eq(r_blending) && eq_constants(l_constants, r_constants),
            _ => false,
        }
    }
}

impl std::hash::Hash for ColorBlend {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            ColorBlend::Logic { op } => {
                op.hash(state);
            }
            ColorBlend::Blending {
                blending,
                write_mask,
                constants,
            } => {
                blending.hash(state);
                write_mask.hash(state);
                hash_constants(constants, state);
            }
            ColorBlend::IndependentBlending {
                blending,
                constants,
            } => {
                blending.hash(state);
                hash_constants(constants, state);
            }
        }
    }
}

impl Default for ColorBlend {
    fn default() -> Self {
        ColorBlend::Blending {
            blending: Some(Blending {
                color_src_factor: BlendFactor::SrcAlpha,
                color_dst_factor: BlendFactor::OneMinusSrcAlpha,
                color_op: BlendOp::Add,
                alpha_src_factor: BlendFactor::One,
                alpha_dst_factor: BlendFactor::OneMinusSrcAlpha,
                alpha_op: BlendOp::Add,
            }),
            write_mask: ComponentMask::RGBA,
            constants: State::Static([0.0; 4]),
        }
    }
}

// NOTE: NaN can be represented by different binary repr, so NaN != NaN
// is always true. This method is required to implement `Eq` trait.
fn eq_constants(lhs: &State<[f32; 4]>, rhs: &State<[f32; 4]>) -> bool {
    match (lhs, rhs) {
        (State::Dynamic, State::Dynamic) => true,
        (State::Static(lhs), State::Static(rhs)) => {
            (*lhs).map(f32::to_bits) == (*rhs).map(f32::to_bits)
        }
        _ => false,
    }
}

fn hash_constants<H: std::hash::Hasher>(constants: &State<[f32; 4]>, state: &mut H) {
    use std::hash::Hash;

    core::mem::discriminant(constants).hash(state);
    match constants {
        &State::Static(constants) => constants.map(f32::to_bits).hash(state),
        State::Dynamic => {}
    }
}

/// Framebuffer logical operations.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum LogicOp {
    /// `0`.
    Clear,
    /// `s & d`
    And,
    /// `s & !d`
    AndReverse,
    /// `s`
    Copy,
    /// `!s & d`
    AndInverted,
    /// `d`
    Noop,
    /// `s ^ d`
    Xor,
    /// `s | d`
    Or,
    /// `!(s | d)`
    Nor,
    /// `!(s ^ d)`
    Equivalent,
    /// `!d`
    Invert,
    /// `s | !d`
    OrReverse,
    /// `!s`
    CopyInverted,
    /// `!s | d`
    OrInverted,
    /// `!(s & d)`
    Nand,
    /// `!0`
    Set,
}

impl FromGfx<LogicOp> for vk::LogicOp {
    fn from_gfx(value: LogicOp) -> Self {
        match value {
            LogicOp::Clear => Self::CLEAR,
            LogicOp::And => Self::AND,
            LogicOp::AndReverse => Self::AND_REVERSE,
            LogicOp::Copy => Self::COPY,
            LogicOp::AndInverted => Self::AND_INVERTED,
            LogicOp::Noop => Self::NO_OP,
            LogicOp::Xor => Self::XOR,
            LogicOp::Or => Self::OR,
            LogicOp::Nor => Self::NOR,
            LogicOp::Equivalent => Self::EQUIVALENT,
            LogicOp::Invert => Self::INVERT,
            LogicOp::OrReverse => Self::OR_REVERSE,
            LogicOp::CopyInverted => Self::COPY_INVERTED,
            LogicOp::OrInverted => Self::OR_INVERTED,
            LogicOp::Nand => Self::NAND,
            LogicOp::Set => Self::SET,
        }
    }
}

/// Framebuffer attachment blending parameters.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Blending {
    pub color_src_factor: BlendFactor,
    pub color_dst_factor: BlendFactor,
    pub color_op: BlendOp,
    pub alpha_src_factor: BlendFactor,
    pub alpha_dst_factor: BlendFactor,
    pub alpha_op: BlendOp,
}

/// Framebuffer blending factors.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum BlendFactor {
    /// Color: `(0.0, 0.0, 0.0)`
    /// Alpha: `0.0`
    Zero,
    /// Color: `(1.0, 1.0, 1.0)`
    /// Alpha: `1.0`
    One,
    /// Color: `(Rs, Gs, Bs)`
    /// Alpha: `As`
    SrcColor,
    /// Color: `(1.0 - Rs, 1.0 - Gs, 1.0 - Bs)`
    /// Alpha: `1.0 - As`
    OneMinusSrcColor,
    /// Color: `(Rd, Gd, Bd)`
    /// Alpha: `Ad`
    DstColor,
    /// Color: `(1.0 - Rd, 1.0 - Gd, 1.0 - Bd)`
    /// Alpha: `1.0 - Ad`
    OneMinusDstColor,
    /// Color: `(As, As, As)`
    /// Alpha: `As`
    SrcAlpha,
    /// Color: `(1.0 - As, 1.0 - As, 1.0 - As)`
    /// Alpha: `1.0 - As`
    OneMinusSrcAlpha,
    /// Color: `(Ad, Ad, Ad)`
    /// Alpha: `Ad`
    DstAlpha,
    /// Color: `(1.0 - Ad, 1.0 - Ad, 1.0 - Ad)`
    /// Alpha: `1.0 - Ad`
    OneMinusDstAlpha,
    /// Color: `(Rc, Gc, Bc)`
    /// Alpha: `Ac`
    ConstantColor,
    /// Color: `(1.0 - Rc, 1.0 - Gc, 1.0 - Bc)`
    /// Alpha: `1.0 - Ac`
    OneMinusConstantColor,
    /// Color: `(Ac, Ac, Ac)`
    /// Alpha: `Ac`
    ConstantAlpha,
    /// Color: `(1.0 - Ac, 1.0 - Ac, 1.0 - Ac)`
    /// Alpha: `1.0 - Ac`
    OneMinusConstantAlpha,
    /// Color: `{let f = min(As, 1.0 - Ad); (f,f,f)}`
    /// Alpha: `1.0`
    SrcAlphaSaturate,
}

impl FromGfx<BlendFactor> for vk::BlendFactor {
    fn from_gfx(value: BlendFactor) -> Self {
        match value {
            BlendFactor::Zero => Self::ZERO,
            BlendFactor::One => Self::ONE,
            BlendFactor::SrcColor => Self::SRC_COLOR,
            BlendFactor::OneMinusSrcColor => Self::ONE_MINUS_SRC_COLOR,
            BlendFactor::DstColor => Self::DST_COLOR,
            BlendFactor::OneMinusDstColor => Self::ONE_MINUS_DST_COLOR,
            BlendFactor::SrcAlpha => Self::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => Self::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstAlpha => Self::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => Self::ONE_MINUS_DST_ALPHA,
            BlendFactor::ConstantColor => Self::CONSTANT_COLOR,
            BlendFactor::OneMinusConstantColor => Self::ONE_MINUS_CONSTANT_COLOR,
            BlendFactor::ConstantAlpha => Self::CONSTANT_ALPHA,
            BlendFactor::OneMinusConstantAlpha => Self::ONE_MINUS_CONSTANT_ALPHA,
            BlendFactor::SrcAlphaSaturate => Self::SRC_ALPHA_SATURATE,
        }
    }
}

/// Framebuffer blending operations.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum BlendOp {
    /// `S * Sf + D * Df`.
    Add,
    /// `S * Sf - D * Df`
    Subtract,
    /// `D * Df - S * Sf`
    ReverseSubtract,
    /// `min(S, D)`
    Min,
    /// `max(S, D)`
    Max,
}

impl FromGfx<BlendOp> for vk::BlendOp {
    fn from_gfx(value: BlendOp) -> Self {
        match value {
            BlendOp::Add => Self::ADD,
            BlendOp::Subtract => Self::SUBTRACT,
            BlendOp::ReverseSubtract => Self::REVERSE_SUBTRACT,
            BlendOp::Min => Self::MIN,
            BlendOp::Max => Self::MAX,
        }
    }
}

bitflags::bitflags! {
    /// Specify which components are written to the framebuffer.
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct ComponentMask: u8 {
        const R = 0b0001;
        const G = 0b0010;
        const B = 0b0100;
        const A = 0b1000;
        const RGB = 0b0111;
        const RGBA = 0b1111;
    }
}

impl FromGfx<ComponentMask> for vk::ColorComponentFlags {
    fn from_gfx(value: ComponentMask) -> Self {
        let mut res = vk::ColorComponentFlags::empty();
        if value.contains(ComponentMask::R) {
            res |= vk::ColorComponentFlags::R;
        }
        if value.contains(ComponentMask::G) {
            res |= vk::ColorComponentFlags::G;
        }
        if value.contains(ComponentMask::B) {
            res |= vk::ColorComponentFlags::B;
        }
        if value.contains(ComponentMask::A) {
            res |= vk::ColorComponentFlags::A;
        }
        res
    }
}

/// A wrapper around a Vulkan graphics pipeline.
///
/// Describes the sequence of operations that take the vertices and textures
/// of meshes all the way to the pixels in the render targets.
pub type GraphicsPipeline = Pipeline<GraphicsPipelineInfo>;

impl std::fmt::Debug for GraphicsPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("GraphicsPipeline")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

// === Compute pipeline ===

/// Structure specifying parameters of a newly created compute pipeline.
#[derive(Debug, Clone)]
pub struct ComputePipelineInfo {
    pub shader: ComputeShader,
    pub layout: PipelineLayout,
}

/// A wrapper around a Vulkan compute pipeline.
pub type ComputePipeline = Pipeline<ComputePipelineInfo>;

impl std::fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("ComputePipeline")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

// === Generic pipeline ===

/// A wrapper around a Vulkan pipeline.
#[repr(transparent)]
pub struct Pipeline<Info> {
    inner: Arc<Inner<Info>>,
}

impl<Info> Pipeline<Info> {
    pub(crate) fn new(handle: vk::Pipeline, info: Info, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.inner.handle
    }

    pub fn info(&self) -> &Info {
        &self.inner.info
    }
}

impl<Info> Clone for Pipeline<Info> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<Info> Eq for Pipeline<Info> {}
impl<Info> PartialEq for Pipeline<Info> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<Info> std::hash::Hash for Pipeline<Info> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner<Info> {
    handle: vk::Pipeline,
    info: Info,
    owner: WeakDevice,
}

impl<Info> Drop for Inner<Info> {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_pipeline(self.handle) }
        }
    }
}
