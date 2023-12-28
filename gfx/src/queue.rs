use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;

use crate::encoder::{CommandBuffer, Encoder};
use crate::surface::SurfaceImage;
use crate::util::{FromGfx, FromVk};

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

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct QueueFlags: u32 {
        const GRAPHICS = 1;
        const COMPUTE = 1 << 1;
        const TRANSFER = 1 << 2;
    }
}

impl QueueFlags {
    pub fn supports_graphics(&self) -> bool {
        self.contains(Self::GRAPHICS)
    }

    pub fn supports_compute(&self) -> bool {
        self.contains(Self::COMPUTE)
    }
}

impl FromVk<vk::QueueFlags> for QueueFlags {
    fn from_vk(flags: vk::QueueFlags) -> Self {
        let mut res = Self::empty();
        if flags.contains(vk::QueueFlags::GRAPHICS) {
            res |= Self::GRAPHICS;
        }
        if flags.contains(vk::QueueFlags::COMPUTE) {
            res |= Self::COMPUTE;
        }
        if flags.contains(vk::QueueFlags::TRANSFER) {
            res |= Self::TRANSFER;
        }
        res
    }
}

impl FromGfx<QueueFlags> for vk::QueueFlags {
    fn from_gfx(flags: QueueFlags) -> Self {
        let mut res = Self::empty();
        if flags.contains(QueueFlags::GRAPHICS) {
            res |= Self::GRAPHICS;
        }
        if flags.contains(QueueFlags::COMPUTE) {
            res |= Self::COMPUTE;
        }
        if flags.contains(QueueFlags::TRANSFER) {
            res |= Self::TRANSFER;
        }
        res
    }
}

pub struct QueueFamily {
    pub capabilities: QueueFlags,
    pub queues: Vec<Queue>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct QueueId {
    pub family: u32,
    pub index: u32,
}

pub struct Queue {
    handle: vk::Queue,
    pool: vk::CommandPool,
    id: QueueId,
    capabilities: QueueFlags,
    command_buffers: Vec<CommandBuffer>,
    device: crate::device::Device,
}

impl Queue {
    pub(crate) fn new(
        handle: vk::Queue,
        family_idx: u32,
        queue_idx: u32,
        capabilities: QueueFlags,
        device: crate::device::Device,
    ) -> Self {
        Self {
            handle,
            pool: vk::CommandPool::null(),
            id: QueueId {
                family: family_idx,
                index: queue_idx,
            },
            capabilities,
            command_buffers: Vec::new(),
            device,
        }
    }

    pub fn id(&self) -> &QueueId {
        &self.id
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.device.logical().queue_wait_idle(self.handle) }?;
        Ok(())
    }

    pub fn create_encoder(&mut self) -> Result<Encoder> {
        let logical = self.device.logical();

        let mut command_buffer = match self.command_buffers.pop() {
            Some(command_buffer) => command_buffer,
            None => {
                if self.pool.is_null() {
                    let info = vk::CommandPoolCreateInfo::builder()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(self.id.family);

                    self.pool = unsafe { logical.create_command_pool(&info, None) }?;
                }

                let handle = {
                    let info = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);

                    let mut buffers = unsafe { logical.allocate_command_buffers(&info) }?;
                    buffers.remove(0)
                };

                CommandBuffer::new(handle, self.id, self.device.clone())
            }
        };

        match command_buffer.begin() {
            Ok(()) => Ok(Encoder::new(command_buffer, self.capabilities)),
            Err(e) => {
                self.command_buffers.push(command_buffer);
                Err(e)
            }
        }
    }

    pub fn present(&mut self, mut image: SurfaceImage<'_>) -> Result<PresentStatus> {
        anyhow::ensure!(
            image
                .supported_families()
                .get(self.id.family as usize)
                .copied()
                .unwrap_or_default(),
            "queue family {} does not support presentation to surface",
            self.id.family
        );

        let [_, signal] = image.wait_signal();

        let res = {
            let logical = self.device.logical();
            unsafe {
                logical.queue_present_khr(
                    self.handle,
                    &vk::PresentInfoKHR::builder()
                        .wait_semaphores(&[signal.handle()])
                        .swapchains(&[image.swapchain_handle()])
                        .image_indices(&[image.index()]),
                )
            }
        };

        image.consume();

        match res {
            Ok(vk::SuccessCode::SUBOPTIMAL_KHR) => Ok(PresentStatus::Suboptimal),
            Ok(_) => Ok(PresentStatus::Ok),
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => Ok(PresentStatus::OutOfDate),
            Err(e) => anyhow::bail!(e),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PresentStatus {
    Ok,
    Suboptimal,
    OutOfDate,
}
