use std::sync::Arc;

use glam::Vec4;
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{Format, FormatChannels, FormatType, ImageLayout, Samples};

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum LoadOp<T = ()> {
    Load,
    Clear(T),
    #[default]
    DontCare,
}

impl<T> From<LoadOp<T>> for vk::AttachmentLoadOp {
    #[inline]
    fn from(value: LoadOp<T>) -> Self {
        match value {
            LoadOp::DontCare => Self::DONT_CARE,
            LoadOp::Clear(_) => Self::CLEAR,
            LoadOp::Load => Self::LOAD,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    #[default]
    DontCare,
}

impl From<StoreOp> for vk::AttachmentStoreOp {
    #[inline]
    fn from(value: StoreOp) -> Self {
        match value {
            StoreOp::Store => Self::STORE,
            StoreOp::DontCare => Self::DONT_CARE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClearValue {
    Color(Vec4),
    DepthStencil(f32, u32),
}

impl ClearValue {
    pub fn to_vk(self, format: Format) -> Option<vk::ClearValue> {
        fn to_uint8(color: f32) -> u8 {
            color.min(0f32).max(u8::max_value() as f32) as u8
        }

        fn to_sint8(color: f32) -> i8 {
            color.min(i8::MIN as f32).max(i8::MAX as f32) as i8
        }

        fn to_uint16(color: f32) -> u16 {
            color.min(0f32).max(u16::MAX as f32) as u16
        }

        fn to_sint16(color: f32) -> i16 {
            color.min(i16::MIN as f32).max(i16::MAX as f32) as i16
        }

        fn to_uint32(color: f32) -> u32 {
            color.min(0f32).max(u32::MAX as f32) as u32
        }

        fn to_sint32(color: f32) -> i32 {
            color.min(i32::MIN as f32).max(i32::MAX as f32) as i32
        }

        fn to_uint64(color: f32) -> u64 {
            color.min(0f32).max(u64::MAX as f32) as u64
        }

        fn to_sint64(color: f32) -> i64 {
            color.min(i64::MIN as f32).max(i64::MAX as f32) as i64
        }

        match self {
            Self::Color(color) => {
                let descr = format.description();
                if matches!(
                    descr.channels,
                    FormatChannels::D | FormatChannels::DS | FormatChannels::S
                ) {
                    return None;
                }

                let color = match (descr.bits, descr.ty) {
                    (8, FormatType::Uint) => vk::ClearColorValue {
                        uint32: [
                            to_uint8(color.x) as _,
                            to_uint8(color.y) as _,
                            to_uint8(color.z) as _,
                            to_uint8(color.w) as _,
                        ],
                    },
                    (8, FormatType::Sint) => vk::ClearColorValue {
                        int32: [
                            to_sint8(color.x) as _,
                            to_sint8(color.y) as _,
                            to_sint8(color.z) as _,
                            to_sint8(color.w) as _,
                        ],
                    },
                    (16, FormatType::Uint) => vk::ClearColorValue {
                        uint32: [
                            to_uint16(color.x) as _,
                            to_uint16(color.y) as _,
                            to_uint16(color.z) as _,
                            to_uint16(color.w) as _,
                        ],
                    },
                    (16, FormatType::Sint) => vk::ClearColorValue {
                        int32: [
                            to_sint16(color.x) as _,
                            to_sint16(color.y) as _,
                            to_sint16(color.z) as _,
                            to_sint16(color.w) as _,
                        ],
                    },
                    (32, FormatType::Uint) => vk::ClearColorValue {
                        uint32: [
                            to_uint32(color.x),
                            to_uint32(color.y),
                            to_uint32(color.z),
                            to_uint32(color.w),
                        ],
                    },
                    (32, FormatType::Sint) => vk::ClearColorValue {
                        int32: [
                            to_sint32(color.x),
                            to_sint32(color.y),
                            to_sint32(color.z),
                            to_sint32(color.w),
                        ],
                    },
                    (64, FormatType::Uint) => vk::ClearColorValue {
                        uint32: [
                            to_uint64(color.x) as _,
                            to_uint64(color.y) as _,
                            to_uint64(color.z) as _,
                            to_uint64(color.w) as _,
                        ],
                    },
                    (64, FormatType::Sint) => vk::ClearColorValue {
                        int32: [
                            to_sint64(color.x) as _,
                            to_sint64(color.y) as _,
                            to_sint64(color.z) as _,
                            to_sint64(color.w) as _,
                        ],
                    },
                    _ => vk::ClearColorValue {
                        float32: [color.x, color.y, color.z, color.w],
                    },
                };
                Some(vk::ClearValue { color })
            }
            Self::DepthStencil(depth, stencil) if !format.is_color() => Some(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearColor(f32, f32, f32, f32);

impl From<ClearColor> for Vec4 {
    #[inline]
    fn from(ClearColor(r, g, b, a): ClearColor) -> Self {
        Self::new(r, g, b, a)
    }
}

impl From<ClearColor> for ClearValue {
    #[inline]
    fn from(value: ClearColor) -> Self {
        Self::Color(value.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearDepth(pub f32);

impl From<ClearDepth> for ClearValue {
    #[inline]
    fn from(value: ClearDepth) -> Self {
        Self::DepthStencil(value.0, 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClearDepthStencil(pub f32, pub u32);

impl From<ClearDepthStencil> for ClearValue {
    #[inline]
    fn from(value: ClearDepthStencil) -> Self {
        Self::DepthStencil(value.0, value.1)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct AttachmentInfo {
    pub format: Format,
    pub samples: Samples,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub initial_layout: Option<ImageLayout>,
    pub final_layout: ImageLayout,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Subpass {
    /// List of color attachment indices and their layouts.
    pub colors: Vec<(u32, ImageLayout)>,
    // Depth attachment index and layout.
    pub depth: Option<(u32, ImageLayout)>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SubpassDependency {
    /// - `Some`: index of the subpass we're dependant on.
    /// - `None`: wait for all of the subpasses within all of the render passes before this one.
    pub src: Option<u32>,
    /// - `Some`: the index to the current subpass, i.e. the one this dependency exists for.
    /// - `None`: all of the subpasses within all of the render passes after this one will depend.
    pub dst: Option<u32>,
    /// Pipeline stages that will be synchronized for the `src`.
    pub src_stages: vk::PipelineStageFlags,
    /// Pipeline stages that will be synchronized for the `dst`.
    pub dst_stages: vk::PipelineStageFlags,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RenderPassInfo {
    pub attachments: Vec<AttachmentInfo>,
    pub subpasses: Vec<Subpass>,
    pub dependencies: Vec<SubpassDependency>,
}

#[derive(Clone)]
pub struct RenderPass {
    inner: Arc<Inner>,
}

impl RenderPass {
    pub(crate) fn new(handle: vk::RenderPass, info: RenderPassInfo, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn handle(&self) -> vk::RenderPass {
        self.inner.handle
    }

    pub fn info(&self) -> &RenderPassInfo {
        &self.inner.info
    }
}

impl std::fmt::Debug for RenderPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("RenderPass")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for RenderPass {}
impl PartialEq for RenderPass {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for RenderPass {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::RenderPass,
    info: RenderPassInfo,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_render_pass(self.handle) }
        }
    }
}
