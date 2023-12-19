use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::queue::QueueId;

#[derive(Default, Debug, Clone, Copy)]
pub enum FenceState {
    #[default]
    Unsignalled,
    Armed {
        queue: QueueId,
        epoch: u64,
    },
    Signalled,
}

pub struct Fence {
    handle: vk::Fence,
    owner: WeakDevice,
    index: usize,
    state: FenceState,
}

impl Fence {
    pub fn new(handle: vk::Fence, owner: WeakDevice, index: usize) -> Self {
        Self {
            handle,
            owner,
            index,
            state: FenceState::Unsignalled,
        }
    }

    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    pub fn state(&self) -> FenceState {
        self.state
    }
}

impl Drop for Fence {
    fn drop(&mut self) {}
}

impl Eq for Fence {}
impl PartialEq for Fence {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl std::hash::Hash for Fence {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.handle.hash(state)
    }
}
