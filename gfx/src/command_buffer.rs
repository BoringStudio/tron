use vulkanalia::prelude::v1_0::*;

use crate::queue::QueueId;

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    queue_id: QueueId,
}

impl CommandBuffer {
    pub fn queue_id(&self) -> QueueId {
        self.queue_id
    }
}
