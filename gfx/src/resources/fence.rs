use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::queue::QueueId;

/// Tracked state of a fence.
#[derive(Default, Debug, Clone, Copy)]
pub enum FenceState {
    #[default]
    Unsignalled,
    Armed {
        queue_id: QueueId,
        epoch: u64,
    },
    Signalled,
}

impl FenceState {
    pub fn is_unsignalled(&self) -> bool {
        matches!(self, Self::Unsignalled)
    }
}

/// A wrapper around a Vulkan fence object.
///
/// Fences are a synchronization primitive that can be used to insert a dependency
/// from a queue to the host.
pub struct Fence {
    handle: vk::Fence,
    owner: WeakDevice,
    state: FenceState,
}

impl Fence {
    pub(crate) fn new(handle: vk::Fence, owner: WeakDevice) -> Self {
        Self {
            handle,
            owner,
            state: FenceState::Unsignalled,
        }
    }

    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    pub fn state(&self) -> FenceState {
        self.state
    }

    pub(crate) fn set_unsignalled(&mut self) -> Result<()> {
        if let FenceState::Armed { .. } = &self.state {
            anyhow::bail!("armed fence cannot be marked as an unsignalled");
        }
        self.state = FenceState::Unsignalled;
        Ok(())
    }

    pub(crate) fn set_armed(
        &mut self,
        queue_id: QueueId,
        epoch: u64,
        device: &crate::device::Device,
    ) -> Result<()> {
        match &self.state {
            FenceState::Unsignalled => {
                self.state = FenceState::Armed { queue_id, epoch };
                Ok(())
            }
            FenceState::Armed { .. } => {
                let signalled = device.update_armed_fence_state(self)?;
                anyhow::ensure!(signalled, "trying to arm an already armed fence");

                // TODO: update previous epoch
                self.state = FenceState::Armed { queue_id, epoch };
                Ok(())
            }
            FenceState::Signalled => {
                anyhow::bail!("arming a signalled fence")
            }
        }
    }

    pub(crate) fn set_signalled(&mut self) -> Result<Option<(QueueId, u64)>> {
        match self.state {
            FenceState::Unsignalled => anyhow::bail!("signalling an unarmed fence"),
            FenceState::Armed {
                queue_id: queue,
                epoch,
            } => {
                self.state = FenceState::Signalled;
                Ok(Some((queue, epoch)))
            }
            FenceState::Signalled => Ok(None),
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            if let FenceState::Armed { .. } = &self.state {
                _ = device.wait_fences(&mut [self], true);
            }

            unsafe { device.destroy_fence(self.handle) };
        }
    }
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
