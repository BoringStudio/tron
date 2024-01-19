use std::sync::{Arc, Mutex};

use arrayvec::ArrayVec;
use bumpalo::Bump;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;

use crate::encoder::{CommandBuffer, CommandBufferLevel, Encoder, PrimaryEncoder};
use crate::resources::{Fence, PipelineStageFlags, Semaphore};
use crate::surface::SurfaceImage;
use crate::types::{DeviceLost, OutOfDeviceMemory, SurfaceLost};
use crate::util::{FromGfx, FromVk, ToGfx, ToVk};

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
#[derive(Clone)]
pub struct Queue {
    inner: Arc<Inner>,
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
            inner: Arc::new(Inner {
                handle,
                submission_mutex: Mutex::default(),
                id: QueueId {
                    family: family_idx,
                    index: queue_idx,
                },
                capabilities,
                cached_buffers: Mutex::new(CachedBuffers::default()),
                device,
            }),
        }
    }

    /// Returns the global queue id.
    pub fn id(&self) -> &QueueId {
        &self.inner.id
    }

    pub fn device(&self) -> &crate::device::Device {
        &self.inner.device
    }

    /// Wait for a queue to become idle.
    pub fn wait_idle(&self) -> Result<(), QueueError> {
        let logical = self.inner.device.logical();
        unsafe { logical.queue_wait_idle(self.inner.handle) }.map_err(|e| match e {
            vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
            vk::ErrorCode::OUT_OF_DEVICE_MEMORY => QueueError::OutOfDeviceMemory(OutOfDeviceMemory),
            vk::ErrorCode::DEVICE_LOST => QueueError::DeviceLost(DeviceLost),
            _ => crate::unexpected_vulkan_error(e),
        })
    }

    /// Begin recording a primary command buffer.
    pub fn create_primary_encoder(&self) -> Result<PrimaryEncoder, OutOfDeviceMemory> {
        let capabilities = self.inner.capabilities;
        self.begin_command_buffer(CommandBufferLevel::Primary)
            .map(|cb| PrimaryEncoder::new(cb, capabilities))
    }

    /// Begin recording a secondary command buffer.
    pub fn create_secondary_encoder(&self) -> Result<Encoder, OutOfDeviceMemory> {
        let capabilities = self.inner.capabilities;
        self.begin_command_buffer(CommandBufferLevel::Secondary)
            .map(|cb| Encoder::new(cb, capabilities))
    }

    /// Submit a set of command buffers to the queue.
    pub fn submit<I>(
        &self,
        wait: &mut [(PipelineStageFlags, &mut Semaphore)],
        command_buffers: I,
        signal: &mut [&mut Semaphore],
        mut fence: Option<&mut Fence>,
        alloc: &mut Bump,
    ) -> Result<(), QueueError>
    where
        I: IntoIterator<Item = CommandBuffer>,
        I::IntoIter: ExactSizeIterator,
    {
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

        let this = self.inner.as_ref();

        if let Some(fence) = fence.as_mut() {
            let epoch = this.device.epochs().next_epoch(this.id);
            fence.set_armed(this.id, epoch, &this.device)?;
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

        let res = {
            let _guard = this.submission_mutex.lock().unwrap();
            unsafe {
                this.device
                    .logical()
                    .queue_submit(this.handle, std::slice::from_ref(&info), fence)
            }
        };
        if let Some(vk::ErrorCode::OUT_OF_HOST_MEMORY) = res.err() {
            crate::out_of_host_memory();
        }

        this.device
            .epochs()
            .submit(this.id, owned_command_buffers.drain(..));

        res.map_err(|e| match e {
            vk::ErrorCode::OUT_OF_DEVICE_MEMORY => QueueError::OutOfDeviceMemory(OutOfDeviceMemory),
            vk::ErrorCode::DEVICE_LOST => QueueError::DeviceLost(DeviceLost),
            _ => crate::unexpected_vulkan_error(e),
        })
    }

    /// Submit a single command buffer to the queue.
    pub fn submit_simple(
        &self,
        command_buffer: CommandBuffer,
        fence: Option<&Fence>,
    ) -> Result<(), QueueError> {
        debug_assert!(
            command_buffer.level() == CommandBufferLevel::Primary,
            "only primary command buffers can be submitted directly to a queue"
        );

        let this = self.inner.as_ref();

        let info = vk::SubmitInfo::builder()
            .command_buffers(&[command_buffer.handle()])
            .build();

        let fence = fence.map(|f| f.handle()).unwrap_or_else(vk::Fence::null);

        let res = {
            let _guard = this.submission_mutex.lock().unwrap();
            unsafe {
                this.device
                    .logical()
                    .queue_submit(this.handle, std::slice::from_ref(&info), fence)
            }
        };
        if let Some(vk::ErrorCode::OUT_OF_HOST_MEMORY) = res.err() {
            crate::out_of_host_memory();
        }

        this.device
            .epochs()
            .submit(this.id, std::iter::once(command_buffer));

        res.map_err(|e| match e {
            vk::ErrorCode::OUT_OF_DEVICE_MEMORY => QueueError::OutOfDeviceMemory(OutOfDeviceMemory),
            vk::ErrorCode::DEVICE_LOST => QueueError::DeviceLost(DeviceLost),
            _ => crate::unexpected_vulkan_error(e),
        })
    }

    /// Present an image to the surface.
    pub fn present(&self, mut image: SurfaceImage<'_>) -> Result<PresentStatus, PresentError> {
        let this = self.inner.as_ref();

        assert!(
            image
                .supported_families()
                .get(this.id.family as usize)
                .copied()
                .unwrap_or_default(),
            "queue family {} does not support presentation to surface",
            this.id.family
        );

        let [_, signal] = image.wait_signal();

        let res = {
            let logical = this.device.logical();

            let _guard = this.submission_mutex.lock().unwrap();
            unsafe {
                logical.queue_present_khr(
                    this.handle,
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
        &self,
        level: CommandBufferLevel,
    ) -> Result<CommandBuffer, OutOfDeviceMemory> {
        let this = self.inner.as_ref();
        let logical = this.device.logical();

        let mut cached = this.cached_buffers.lock().unwrap();
        let cached = &mut *cached;

        let command_buffers = match level {
            CommandBufferLevel::Primary => &mut cached.primary_command_buffers,
            CommandBufferLevel::Secondary => &mut cached.secondary_command_buffers,
        };

        let mut command_buffer = match command_buffers.pop() {
            Some(command_buffer) => command_buffer,
            None => {
                let queue_family = this.id.family;

                if cached.pool.is_null() {
                    let info = vk::CommandPoolCreateInfo::builder()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(queue_family);

                    cached.pool = unsafe { logical.create_command_pool(&info, None) }
                        .map_err(OutOfDeviceMemory::on_creation)?;
                }

                let handle = {
                    let info = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(cached.pool)
                        .level(level.to_vk())
                        .command_buffer_count(1);

                    let mut buffers = unsafe { logical.allocate_command_buffers(&info) }
                        .map_err(OutOfDeviceMemory::on_creation)?;
                    buffers.remove(0)
                };

                tracing::debug!(command_buffer = ?handle, ?level, "created command buffer");

                CommandBuffer::new(handle, queue_family, level, this.device.clone())
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

    fn restore_command_buffers(&self) -> Result<(), OutOfDeviceMemory> {
        let this = self.inner.as_ref();
        let logical = this.device.logical();

        let mut cached = this.cached_buffers.lock().unwrap();
        let cached = &mut *cached;

        let primary_offset = cached.primary_command_buffers.len();
        let secondaty_offset = cached.secondary_command_buffers.len();

        this.device.epochs().drain_free_command_buffers(
            this.id,
            &mut cached.primary_command_buffers,
            &mut cached.secondary_command_buffers,
        );

        let primary = &cached.primary_command_buffers[primary_offset..];
        let secondary = &cached.secondary_command_buffers[secondaty_offset..];

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

impl std::fmt::Debug for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("Queue")
                .field("id", &self.inner.id)
                .field("capabilities", &self.inner.capabilities)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for Queue {}
impl PartialEq for Queue {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for Queue {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::Queue,
    submission_mutex: Mutex<()>,
    id: QueueId,
    cached_buffers: Mutex<CachedBuffers>,
    capabilities: QueueFlags,
    device: crate::device::Device,
}

#[derive(Default)]
struct CachedBuffers {
    pool: vk::CommandPool,
    primary_command_buffers: Vec<CommandBuffer>,
    secondary_command_buffers: Vec<CommandBuffer>,
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
