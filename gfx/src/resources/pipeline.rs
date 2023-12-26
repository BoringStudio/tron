use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{ComputeShader, PipelineLayout};

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
