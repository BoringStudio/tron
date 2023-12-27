use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::util::{FromGfx, ToVk};

#[derive(Debug, Default, Clone, Copy)]
pub struct SamplerInfo {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_mode: MipmapMode,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub max_anisotropy: Option<f32>,
    pub compare_op: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: BorderColor,
    pub unnormalized_coordinates: bool,
}

impl SamplerInfo {
    pub fn simple_nearest() -> Self {
        Self::default()
    }

    pub fn simple_linear() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_mode: MipmapMode::Linear,
            ..Default::default()
        }
    }
}

impl Eq for SamplerInfo {}
impl PartialEq for SamplerInfo {
    fn eq(&self, other: &Self) -> bool {
        self.min_filter.eq(&other.min_filter)
            && self.mag_filter.eq(&other.mag_filter)
            && self.mipmap_mode.eq(&other.mipmap_mode)
            && self.address_mode_u.eq(&other.address_mode_u)
            && self.address_mode_v.eq(&other.address_mode_v)
            && self.address_mode_w.eq(&other.address_mode_w)
            && f32::to_bits(self.mip_lod_bias).eq(&f32::to_bits(other.mip_lod_bias))
            && self
                .max_anisotropy
                .map(f32::to_bits)
                .eq(&other.max_anisotropy.map(f32::to_bits))
            && self.compare_op.eq(&other.compare_op)
            && f32::to_bits(self.min_lod).eq(&f32::to_bits(other.min_lod))
            && f32::to_bits(self.max_lod).eq(&f32::to_bits(other.max_lod))
            && self.border_color.eq(&other.border_color)
            && self
                .unnormalized_coordinates
                .eq(&other.unnormalized_coordinates)
    }
}

impl std::hash::Hash for SamplerInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.mag_filter.hash(state);
        self.min_filter.hash(state);
        self.mipmap_mode.hash(state);
        self.address_mode_u.hash(state);
        self.address_mode_v.hash(state);
        self.address_mode_w.hash(state);
        f32::to_bits(self.mip_lod_bias).hash(state);
        self.max_anisotropy.map(f32::to_bits).hash(state);
        self.compare_op.hash(state);
        f32::to_bits(self.min_lod).hash(state);
        f32::to_bits(self.max_lod).hash(state);
        self.border_color.hash(state);
        self.unnormalized_coordinates.hash(state);
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Filter {
    #[default]
    Nearest,
    Linear,
}

impl FromGfx<Filter> for vk::Filter {
    fn from_gfx(value: Filter) -> Self {
        match value {
            Filter::Nearest => Self::NEAREST,
            Filter::Linear => Self::LINEAR,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MipmapMode {
    #[default]
    Nearest,
    Linear,
}

impl FromGfx<MipmapMode> for vk::SamplerMipmapMode {
    fn from_gfx(value: MipmapMode) -> Self {
        match value {
            MipmapMode::Nearest => Self::NEAREST,
            MipmapMode::Linear => Self::LINEAR,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    #[default]
    ClampToEdge,
    ClampToBorder,
    MirrorClampToEdge,
}

impl FromGfx<SamplerAddressMode> for vk::SamplerAddressMode {
    fn from_gfx(value: SamplerAddressMode) -> Self {
        match value {
            SamplerAddressMode::Repeat => Self::REPEAT,
            SamplerAddressMode::MirroredRepeat => Self::MIRRORED_REPEAT,
            SamplerAddressMode::ClampToEdge => Self::CLAMP_TO_EDGE,
            SamplerAddressMode::ClampToBorder => Self::CLAMP_TO_BORDER,
            SamplerAddressMode::MirrorClampToEdge => Self::MIRROR_CLAMP_TO_EDGE,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CompareOp {
    /// Never passes.
    Never,
    /// Passes if fragment's depth is less than stored.
    Less,
    /// Passes if fragment's depth is equal to stored.
    Equal,
    /// Passes if fragment's depth is less than or equal to stored.
    LessOrEqual,
    /// Passes if fragment's depth is greater than stored.
    Greater,
    /// Passes if fragment's depth is not equal to stored.
    NotEqual,
    /// Passes if fragment's depth is greater than or equal to stored.
    GreaterOrEqual,
    /// Always passes.
    Always,
}

impl FromGfx<CompareOp> for vk::CompareOp {
    fn from_gfx(value: CompareOp) -> Self {
        match value {
            CompareOp::Never => Self::NEVER,
            CompareOp::Less => Self::LESS,
            CompareOp::Equal => Self::EQUAL,
            CompareOp::LessOrEqual => Self::LESS_OR_EQUAL,
            CompareOp::Greater => Self::GREATER,
            CompareOp::NotEqual => Self::NOT_EQUAL,
            CompareOp::GreaterOrEqual => Self::GREATER_OR_EQUAL,
            CompareOp::Always => Self::ALWAYS,
        }
    }
}

impl FromGfx<Option<CompareOp>> for vk::CompareOp {
    fn from_gfx(value: Option<CompareOp>) -> Self {
        match value {
            Some(value) => value.to_vk(),
            None => Self::NEVER,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum BorderColor {
    #[default]
    FloatTransparentBlack,
    IntTransparentBlack,
    FloatOpaqueBlack,
    IntOpaqueBlack,
    FloatOpaqueWhite,
    IntOpaqueWhite,
}

impl FromGfx<BorderColor> for vk::BorderColor {
    fn from_gfx(value: BorderColor) -> Self {
        match value {
            BorderColor::FloatTransparentBlack => Self::FLOAT_TRANSPARENT_BLACK,
            BorderColor::IntTransparentBlack => Self::INT_TRANSPARENT_BLACK,
            BorderColor::FloatOpaqueBlack => Self::FLOAT_OPAQUE_BLACK,
            BorderColor::IntOpaqueBlack => Self::INT_OPAQUE_BLACK,
            BorderColor::FloatOpaqueWhite => Self::FLOAT_OPAQUE_WHITE,
            BorderColor::IntOpaqueWhite => Self::INT_OPAQUE_WHITE,
        }
    }
}

#[derive(Clone)]
pub struct Sampler {
    inner: Arc<Inner>,
}

impl Sampler {
    pub(crate) fn new(handle: vk::Sampler, info: SamplerInfo, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn handle(&self) -> vk::Sampler {
        self.inner.handle
    }

    pub fn info(&self) -> &SamplerInfo {
        &self.inner.info
    }
}

struct Inner {
    handle: vk::Sampler,
    info: SamplerInfo,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_sampler(self.handle) }
        }
    }
}
