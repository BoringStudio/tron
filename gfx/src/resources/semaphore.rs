use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;

pub struct Semaphore {
    handle: vk::Semaphore,
    owner: WeakDevice,
}

impl Semaphore {
    pub fn new(handle: vk::Semaphore, owner: WeakDevice) -> Self {
        Self { handle, owner }
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_semaphore(self.handle) };
        }
    }
}

impl Eq for Semaphore {}
impl PartialEq for Semaphore {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl std::hash::Hash for Semaphore {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.handle.hash(state)
    }
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("Semaphore")
                .field("handle", &self.handle)
                .field("owner", &self.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.handle, f)
        }
    }
}
