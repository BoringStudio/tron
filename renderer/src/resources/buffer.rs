use vulkanalia::prelude::v1_0::*;

use crate::types::DeviceAddress;

pub struct Buffer {
    handle: vk::Buffer,
    memory_usage: gpu_alloc::MemoryPropertyFlags,
    address: Option<DeviceAddress>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct BufferInfo {
    pub align: u64,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
}
