use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::{Buffer, Format};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BufferViewInfo {
    pub buffer: Buffer,
    pub format: Format,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BufferRange {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
}

impl BufferRange {
    pub fn whole(buffer: Buffer) -> Self {
        Self {
            offset: 0,
            size: buffer.info().size,
            buffer,
        }
    }
}

impl From<Buffer> for BufferRange {
    #[inline]
    fn from(buffer: Buffer) -> Self {
        Self::whole(buffer)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct BufferView {
    inner: Arc<Inner>,
}

impl BufferView {
    pub(crate) fn new(handle: vk::BufferView, info: BufferViewInfo, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn handle(&self) -> vk::BufferView {
        self.inner.handle
    }

    pub fn info(&self) -> &BufferViewInfo {
        &self.inner.info
    }
}

impl std::fmt::Debug for BufferView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("BufferView")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for BufferView {}
impl PartialEq for BufferView {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for BufferView {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::BufferView,
    info: BufferViewInfo,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_buffer_view(self.handle) }
        }
    }
}
