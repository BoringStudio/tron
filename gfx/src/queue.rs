use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

pub trait QueuesQuery {
    type QueryState;
    type Query: AsRef<[(usize, usize)]>;
    type Queues;

    fn query(
        self,
        families: &[vk::QueueFamilyProperties],
    ) -> Result<(Self::Query, Self::QueryState)>;
    fn collect(state: Self::QueryState, families: Vec<QueueFamily>) -> Self::Queues;
}

#[derive(Debug, Clone, Copy)]
pub struct SingleQueueQuery(vk::QueueFlags);

impl SingleQueueQuery {
    pub const COMPUTE: Self = Self(vk::QueueFlags::COMPUTE);
    pub const GRAPHICS: Self = Self(vk::QueueFlags::GRAPHICS);
    pub const TRANSFER: Self = Self(vk::QueueFlags::TRANSFER);
}

impl QueuesQuery for SingleQueueQuery {
    type QueryState = ();
    type Query = [(usize, usize); 1];
    type Queues = Queue;

    fn query(
        self,
        families: &[vk::QueueFamilyProperties],
    ) -> Result<(Self::Query, Self::QueryState)> {
        for (index, family) in families.iter().enumerate() {
            if family.queue_count > 0 && family.queue_flags.contains(self.0) {
                return Ok(([(index, 1)], ()));
            }
        }
        anyhow::bail!("queue not found {:?}", self.0);
    }

    fn collect(_state: Self::QueryState, mut families: Vec<QueueFamily>) -> Self::Queues {
        families.remove(0).queues.remove(0)
    }
}

pub struct QueueFamily {
    pub capabilities: vk::QueueFlags,
    pub queues: Vec<Queue>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct QueueId {
    pub family: u32,
    pub index: u32,
}

pub struct Queue {
    handle: vk::Queue,
    id: QueueId,
    _capabilities: vk::QueueFlags,
    device: crate::device::Device,

    _pool: vk::CommandPool,
}

impl Queue {
    pub fn new(
        handle: vk::Queue,
        family_idx: u32,
        queue_idx: u32,
        capabilities: vk::QueueFlags,
        device: crate::device::Device,
    ) -> Self {
        Self {
            handle,
            id: QueueId {
                family: family_idx,
                index: queue_idx,
            },
            _capabilities: capabilities,
            device,
            _pool: vk::CommandPool::null(),
        }
    }

    pub fn id(&self) -> &QueueId {
        &self.id
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.device.logical().queue_wait_idle(self.handle) }?;
        Ok(())
    }
}
