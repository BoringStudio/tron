use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::DescriptorSetLayout;

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct PipelineLayoutInfo {
    pub sets: Vec<DescriptorSetLayout>,
    pub push_constants: Vec<PushConstant>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PushConstant {
    pub stages: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone)]
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
