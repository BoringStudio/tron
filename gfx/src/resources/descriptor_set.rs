use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::{AllocatedDescriptorSet, WeakDevice};
use crate::resources::DescriptorSetLayout;

#[derive(Debug, Clone)]
pub struct DescriptorSetInfo {
    pub layout: DescriptorSetLayout,
}

#[derive(Clone)]
pub struct DescriptorSet {
    inner: Arc<Inner>,
}

impl DescriptorSet {
    pub(crate) fn new(allocated: AllocatedDescriptorSet, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner { allocated, owner }),
        }
    }

    pub fn handle(&self) -> vk::DescriptorSet {
        self.inner.allocated.handle()
    }
}

impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("DescriptorSet")
                .field("handle", &self.inner.allocated.handle())
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.allocated.handle(), f)
        }
    }
}

impl Eq for DescriptorSet {}
impl PartialEq for DescriptorSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for DescriptorSet {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    allocated: AllocatedDescriptorSet,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_descriptor_set(&self.allocated) }
        }
    }
}
