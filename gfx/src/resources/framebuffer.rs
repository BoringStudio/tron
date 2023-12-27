use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{ImageView, RenderPass};

#[derive(Debug, Clone, Hash)]
pub struct FramebufferInfo {
    pub render_pass: RenderPass,
    pub attachments: Vec<ImageView>,
    pub extent: glam::UVec2,
}

#[derive(Clone)]
pub struct Framebuffer {
    inner: Arc<Inner>,
}

impl Framebuffer {
    pub(crate) fn new(handle: vk::Framebuffer, info: FramebufferInfo, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn handle(&self) -> vk::Framebuffer {
        self.inner.handle
    }

    pub fn info(&self) -> &FramebufferInfo {
        &self.inner.info
    }
}

impl std::fmt::Debug for Framebuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("Framebuffer")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for Framebuffer {}
impl PartialEq for Framebuffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for Framebuffer {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::Framebuffer,
    info: FramebufferInfo,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_framebuffer(self.handle) }
        }
    }
}
