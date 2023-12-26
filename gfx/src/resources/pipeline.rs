use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{
    CompareOp, ComputeShader, Format, FragmentShader, PipelineLayout, RenderPass, VertexShader,
};
use crate::types::State;

// === Graphics pipeline ===

#[derive(Debug, Clone)]
pub struct GraphicsPipelineInfo {
    pub descr: GraphicsPipelineDescr,
    pub rendering: GraphicsPipelineRenderingInfo,
}

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

#[derive(Debug, Clone)]
pub struct GraphicsPipelineRenderingInfo {
    pub render_pass: RenderPass,
    pub subpass: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VertexInputBinding {
    pub rate: VertexInputRate,
    pub stride: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VertexInputAttribute {
    pub location: u32,
    pub format: Format,
    pub binding: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum VertexInputRate {
    Vertex,
    Instance,
}

impl From<VertexInputRate> for vk::VertexInputRate {
    fn from(value: VertexInputRate) -> Self {
        match value {
            VertexInputRate::Vertex => Self::VERTEX,
            VertexInputRate::Instance => Self::INSTANCE,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    #[default]
    TriangleList,
    TriangleStrip,
    TriangleFan,
}

impl From<PrimitiveTopology> for vk::PrimitiveTopology {
    fn from(value: PrimitiveTopology) -> Self {
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

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum FrontFace {
    #[default]
    CW,
    CCW,
}

impl From<FrontFace> for vk::FrontFace {
    fn from(value: FrontFace) -> Self {
        match value {
            FrontFace::CW => Self::CLOCKWISE,
            FrontFace::CCW => Self::COUNTER_CLOCKWISE,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CullMode {
    Front,
    Back,
    FrontAndBack,
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(value: CullMode) -> Self {
        match value {
            CullMode::Front => Self::FRONT,
            CullMode::Back => Self::BACK,
            CullMode::FrontAndBack => Self::FRONT_AND_BACK,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
    Point,
}

impl From<PolygonMode> for vk::PolygonMode {
    fn from(value: PolygonMode) -> Self {
        match value {
            PolygonMode::Fill => Self::FILL,
            PolygonMode::Line => Self::LINE,
            PolygonMode::Point => Self::POINT,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DepthTest {
    pub compare: CompareOp,
    pub write: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct StencilTests {
    pub front: StencilTest,
    pub back: StencilTest,
}

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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementAndClamp,
    DecrementAndClamp,
    Invert,
    IncrementAndWrap,
    DecrementAndWrap,
}

impl From<StencilOp> for vk::StencilOp {
    fn from(value: StencilOp) -> Self {
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

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub min: f32,
    pub max: f32,
}

impl Eq for Bounds {}
impl PartialEq for Bounds {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        f32::to_bits(self.min) == f32::to_bits(other.min)
            && f32::to_bits(self.max) == f32::to_bits(other.max)
    }
}

impl std::hash::Hash for Bounds {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(f32::to_bits(self.min));
        state.write_u32(f32::to_bits(self.max));
    }
}

#[derive(Debug, Clone)]
pub enum ColorBlend {
    Logic {
        op: LogicOp,
    },
    Blending {
        blending: Option<Blending>,
        write_mask: ComponentMask,
        constants: State<[f32; 4]>,
    },
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

impl From<LogicOp> for vk::LogicOp {
    fn from(value: LogicOp) -> Self {
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Blending {
    pub color_src_factor: BlendFactor,
    pub color_dst_factor: BlendFactor,
    pub color_op: BlendOp,
    pub alpha_src_factor: BlendFactor,
    pub alpha_dst_factor: BlendFactor,
    pub alpha_op: BlendOp,
}

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

impl From<BlendFactor> for vk::BlendFactor {
    fn from(value: BlendFactor) -> Self {
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

impl From<BlendOp> for vk::BlendOp {
    fn from(value: BlendOp) -> Self {
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

impl From<ComponentMask> for vk::ColorComponentFlags {
    fn from(value: ComponentMask) -> Self {
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

#[derive(Debug, Clone)]
pub struct ComputePipelineInfo {
    pub shader: ComputeShader,
    pub layout: PipelineLayout,
}

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
