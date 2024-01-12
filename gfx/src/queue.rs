use arrayvec::ArrayVec;
use bumpalo::Bump;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;

use crate::encoder::{CommandBuffer, CommandBufferLevel, Encoder, PrimaryEncoder};
use crate::resources::{Fence, PipelineStageFlags, Semaphore};
use crate::surface::SurfaceImage;
use crate::types::{DeviceLost, OutOfDeviceMemory, SurfaceLost};
use crate::util::{DeallocOnDrop, FromGfx, FromVk, ToGfx, ToVk};

/// A query for a set of queues.
pub trait QueuesQuery {
    type QueryState;
    type Query: AsRef<[(usize, usize)]>;
    type Queues;
    type Error;

    fn query(
        self,
        families: &[vk::QueueFamilyProperties],
    ) -> Result<(Self::Query, Self::QueryState), Self::Error>;
    fn collect(state: Self::QueryState, families: Vec<QueueFamily>) -> Self::Queues;
}

/// Single queue query.
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
    type Error = QueueNotFound;

    fn query(
        self,
        families: &[vk::QueueFamilyProperties],
    ) -> Result<(Self::Query, Self::QueryState), Self::Error> {
        for (index, family) in families.iter().enumerate() {
            if family.queue_count > 0 && family.queue_flags.contains(self.0) {
                return Ok(([(index, 1)], ()));
            }
        }
        Err(QueueNotFound {
            capabilities: self.0.to_gfx(),
        })
    }

    fn collect(_state: Self::QueryState, mut families: Vec<QueueFamily>) -> Self::Queues {
        families.remove(0).queues.remove(0)
    }
}

bitflags::bitflags! {
    /// Queue capabilities.
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

/// A collection of queues.
pub struct QueueFamily {
    pub capabilities: QueueFlags,
    pub queues: Vec<Queue>,
}

/// A global queue id.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct QueueId {
    pub family: u32,
    pub index: u32,
}

/// A wrapper around a Vulkan queue.
pub struct Queue {
    handle: vk::Queue,
    pool: vk::CommandPool,
    id: QueueId,
    capabilities: QueueFlags,
    primary_command_buffers: Vec<CommandBuffer>,
    secondary_command_buffers: Vec<CommandBuffer>,
    device: crate::device::Device,
    alloc: Bump,
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
            primary_command_buffers: Vec::new(),
            secondary_command_buffers: Vec::new(),
            device,
            alloc: Bump::new(),
        }
    }

    /// Returns the global queue id.
    pub fn id(&self) -> &QueueId {
        &self.id
    }

    /// Wait for a queue to become idle.
    pub fn wait_idle(&self) -> Result<(), QueueError> {
        unsafe { self.device.logical().queue_wait_idle(self.handle) }.map_err(|e| match e {
            vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
            vk::ErrorCode::OUT_OF_DEVICE_MEMORY => QueueError::OutOfDeviceMemory(OutOfDeviceMemory),
            vk::ErrorCode::DEVICE_LOST => QueueError::DeviceLost(DeviceLost),
            _ => crate::unexpected_vulkan_error(e),
        })
    }

    /// Begin recording a primary command buffer.
    pub fn create_primary_encoder(&mut self) -> Result<PrimaryEncoder, OutOfDeviceMemory> {
        self.begin_command_buffer(CommandBufferLevel::Primary)
            .map(|cb| PrimaryEncoder::new(cb, self.capabilities))
    }

    /// Begin recording a secondary command buffer.
    pub fn create_secondary_encoder(&mut self) -> Result<Encoder, OutOfDeviceMemory> {
        self.begin_command_buffer(CommandBufferLevel::Secondary)
            .map(|cb| Encoder::new(cb, self.capabilities))
    }

    /// Submit a set of command buffers to the queue.
    pub fn submit<I>(
        &mut self,
        wait: &mut [(PipelineStageFlags, &mut Semaphore)],
        command_buffers: I,
        signal: &mut [&mut Semaphore],
        mut fence: Option<&mut Fence>,
    ) -> Result<(), QueueError>
    where
        I: IntoIterator<Item = CommandBuffer>,
        I::IntoIter: ExactSizeIterator,
    {
        let alloc = DeallocOnDrop(&mut self.alloc);

        let owned_command_buffers = alloc.alloc_with(ArrayVec::<_, 64>::new);
        let command_buffers =
            alloc.alloc_slice_fill_iter(command_buffers.into_iter().map(|command_buffer| {
                debug_assert!(
                    command_buffer.level() == CommandBufferLevel::Primary,
                    "only primary command buffers can be submitted directly to a queue"
                );

                let handle = command_buffer.handle();
                owned_command_buffers.push(command_buffer);
                handle
            }));

        if let Some(fence) = fence.as_mut() {
            let epoch = self.device.epochs().next_epoch(self.id);
            fence.set_armed(self.id, epoch, &self.device)?;
        }

        let wait_stages = alloc.alloc_slice_fill_iter(
            wait.iter()
                .map(|(stage, _)| vk::PipelineStageFlags::from_gfx(*stage)),
        );
        let wait_semaphores =
            alloc.alloc_slice_fill_iter(wait.iter().map(|(_, semaphore)| semaphore.handle()));
        let signal_semaphores =
            alloc.alloc_slice_fill_iter(signal.iter().map(|semaphore| semaphore.handle()));

        let info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .build();

        let fence = fence.map(|f| f.handle()).unwrap_or_else(vk::Fence::null);

        let res = unsafe {
            self.device
                .logical()
                .queue_submit(self.handle, std::slice::from_ref(&info), fence)
        };
        if let Some(vk::ErrorCode::OUT_OF_HOST_MEMORY) = res.err() {
            crate::out_of_host_memory();
        }

        self.device
            .epochs()
            .submit(self.id, owned_command_buffers.drain(..));

        res.map_err(|e| match e {
            vk::ErrorCode::OUT_OF_DEVICE_MEMORY => QueueError::OutOfDeviceMemory(OutOfDeviceMemory),
            vk::ErrorCode::DEVICE_LOST => QueueError::DeviceLost(DeviceLost),
            _ => crate::unexpected_vulkan_error(e),
        })
    }

    /// Submit a single command buffer to the queue.
    pub fn submit_simple(
        &mut self,
        command_buffer: CommandBuffer,
        fence: Option<&Fence>,
    ) -> Result<(), QueueError> {
        debug_assert!(
            command_buffer.level() == CommandBufferLevel::Primary,
            "only primary command buffers can be submitted directly to a queue"
        );

        let info = vk::SubmitInfo::builder()
            .command_buffers(&[command_buffer.handle()])
            .build();

        let fence = fence.map(|f| f.handle()).unwrap_or_else(vk::Fence::null);

        let res = unsafe {
            self.device
                .logical()
                .queue_submit(self.handle, std::slice::from_ref(&info), fence)
        };
        if let Some(vk::ErrorCode::OUT_OF_HOST_MEMORY) = res.err() {
            crate::out_of_host_memory();
        }

        self.device
            .epochs()
            .submit(self.id, std::iter::once(command_buffer));

        res.map_err(|e| match e {
            vk::ErrorCode::OUT_OF_DEVICE_MEMORY => QueueError::OutOfDeviceMemory(OutOfDeviceMemory),
            vk::ErrorCode::DEVICE_LOST => QueueError::DeviceLost(DeviceLost),
            _ => crate::unexpected_vulkan_error(e),
        })
    }

    /// Present an image to the surface.
    pub fn present(&mut self, mut image: SurfaceImage<'_>) -> Result<PresentStatus, PresentError> {
        assert!(
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
        if let Some(vk::ErrorCode::OUT_OF_HOST_MEMORY) = res.err() {
            crate::out_of_host_memory();
        }

        image.consume();

        self.restore_command_buffers()?;

        match res {
            Ok(vk::SuccessCode::SUBOPTIMAL_KHR) => Ok(PresentStatus::Suboptimal),
            Ok(_) => Ok(PresentStatus::Ok),
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => Ok(PresentStatus::OutOfDate),
            Err(e) => Err(match e {
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => PresentError::from(OutOfDeviceMemory),
                vk::ErrorCode::DEVICE_LOST => PresentError::from(DeviceLost),
                vk::ErrorCode::SURFACE_LOST_KHR => PresentError::from(SurfaceLost),
                _ => crate::unexpected_vulkan_error(e),
            }),
        }
    }

    fn begin_command_buffer(
        &mut self,
        level: CommandBufferLevel,
    ) -> Result<CommandBuffer, OutOfDeviceMemory> {
        let logical = self.device.logical();

        let command_buffers = match level {
            CommandBufferLevel::Primary => &mut self.primary_command_buffers,
            CommandBufferLevel::Secondary => &mut self.secondary_command_buffers,
        };

        let mut command_buffer = match command_buffers.pop() {
            Some(command_buffer) => command_buffer,
            None => {
                if self.pool.is_null() {
                    let info = vk::CommandPoolCreateInfo::builder()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(self.id.family);

                    self.pool = unsafe { logical.create_command_pool(&info, None) }
                        .map_err(OutOfDeviceMemory::on_creation)?;
                }

                let handle = {
                    let info = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.pool)
                        .level(level.to_vk())
                        .command_buffer_count(1);

                    let mut buffers = unsafe { logical.allocate_command_buffers(&info) }
                        .map_err(OutOfDeviceMemory::on_creation)?;
                    buffers.remove(0)
                };

                tracing::debug!(command_buffer = ?handle, ?level, "created command buffer");

                CommandBuffer::new(handle, self.id, level, self.device.clone())
            }
        };

        debug_assert!(command_buffer.references().is_empty());
        debug_assert!(command_buffer.secondary_buffers().is_empty());

        match command_buffer.begin() {
            Ok(()) => Ok(command_buffer),
            Err(e) => {
                command_buffers.push(command_buffer);
                Err(e)
            }
        }
    }

    fn restore_command_buffers(&mut self) -> Result<(), OutOfDeviceMemory> {
        let logical = self.device.logical();

        let primary_offset = self.primary_command_buffers.len();
        let secondaty_offset = self.secondary_command_buffers.len();
        self.device.epochs().drain_free_command_buffers(
            self.id,
            &mut self.primary_command_buffers,
            &mut self.secondary_command_buffers,
        );

        let primary = &self.primary_command_buffers[primary_offset..];
        let secondary = &self.secondary_command_buffers[secondaty_offset..];

        for cb in primary.iter().chain(secondary) {
            unsafe {
                logical.reset_command_buffer(
                    cb.handle(),
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
            }
            .map_err(|e| match e {
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => OutOfDeviceMemory,
                _ => crate::unexpected_vulkan_error(e),
            })?;
        }

        Ok(())
    }
}

/// The result of a present operation.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PresentStatus {
    Ok,
    Suboptimal,
    OutOfDate,
}

/// Queue presentation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PresentError {
    #[error(transparent)]
    OutOfDeviceMemory(#[from] OutOfDeviceMemory),
    #[error(transparent)]
    DeviceLost(#[from] DeviceLost),
    #[error(transparent)]
    SurfaceLost(#[from] SurfaceLost),
}

/// Runtime queue error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum QueueError {
    #[error(transparent)]
    OutOfDeviceMemory(#[from] OutOfDeviceMemory),
    #[error(transparent)]
    DeviceLost(#[from] DeviceLost),
}

/// Queue not found error.
#[derive(Debug, Clone, thiserror::Error)]
#[error("no queue found with capabilities {capabilities:?}")]
pub struct QueueNotFound {
    pub capabilities: QueueFlags,
}
