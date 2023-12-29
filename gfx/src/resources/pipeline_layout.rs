use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{DescriptorSetLayout, ShaderStageFlags};
use crate::util::{FromGfx, ToVk};

/// Structure specifying the parameters of a newly created pipeline layout object.
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct PipelineLayoutInfo {
    pub sets: Vec<DescriptorSetLayout>,
    pub push_constants: Vec<PushConstant>,
}

/// Structure specifying a push constant range.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PushConstant {
    pub stages: ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

impl FromGfx<PushConstant> for vk::PushConstantRange {
    fn from_gfx(value: PushConstant) -> Self {
        Self {
            stage_flags: value.stages.to_vk(),
            offset: value.offset,
            size: value.size,
        }
    }
}

/// A wrapper around a Vulkan pipeline layout object.
///
/// Access to descriptor sets from a pipeline is accomplished through a pipeline layout.
/// Zero or more descriptor set layouts and zero or more push constant ranges are combined
/// to form a pipeline layout object describing the complete set of resources that can be
/// accessed by a pipeline. The pipeline layout represents a sequence of descriptor sets
/// with each having a specific layout. This sequence of layouts is used to determine the
/// interface between shader stages and shader resources. Each pipeline is created using
/// a pipeline layout.
#[derive(Clone)]
#[repr(transparent)]
pub struct PipelineLayout {
    inner: Arc<Inner>,
}

impl PipelineLayout {
    pub(crate) fn new(
        handle: vk::PipelineLayout,
        info: PipelineLayoutInfo,
        owner: WeakDevice,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn handle(&self) -> vk::PipelineLayout {
        self.inner.handle
    }

    pub fn info(&self) -> &PipelineLayoutInfo {
        &self.inner.info
    }
}

impl std::fmt::Debug for PipelineLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("PipelineLayout")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for PipelineLayout {}
impl PartialEq for PipelineLayout {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for PipelineLayout {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::PipelineLayout,
    info: PipelineLayoutInfo,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_pipeline_layout(self.handle) }
        }
    }
}
