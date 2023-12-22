use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{ImageLayout, Samples};

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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct AttachmentInfo {
    pub format: vk::Format,
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
