use std::cell::UnsafeCell;
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use gpu_alloc::MemoryBlock;
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::types::DeviceAddress;
use crate::util::FromGfx;

/// Type of index buffer indices.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum IndexType {
    U16,
    U32,
}

impl IndexType {
    pub const fn from_vk(value: vk::IndexType) -> Option<Self> {
        match value {
            vk::IndexType::UINT16 => Some(Self::U16),
            vk::IndexType::UINT32 => Some(Self::U32),
            _ => None,
        }
    }

    pub const fn index_size(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

impl FromGfx<IndexType> for vk::IndexType {
    fn from_gfx(value: IndexType) -> Self {
        match value {
            IndexType::U16 => vk::IndexType::UINT16,
            IndexType::U32 => vk::IndexType::UINT32,
        }
    }
}

/// Structure specifying the parameters of a newly created buffer object.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferInfo {
    pub align_mask: usize,
    pub size: usize,
    pub usage: BufferUsage,
}

bitflags::bitflags! {
    /// Bitmask specifying allowed usage of a buffer.
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        /// The buffer can be used as the source of a transfer command.
        const TRANSFER_SRC = 1;
        /// The buffer can be used as the destination of a transfer command.
        const TRANSFER_DST = 1 << 1;
        /// The buffer can be used to create a [`BufferView`] suitable for occupying a
        /// [`DescriptorSet`] slot of type [`DescriptorType::UniformTexelBuffer`].
        ///
        /// [`BufferView`]: crate::BufferView
        /// [`DescriptorSet`]: crate::DescriptorSet
        /// [`DescriptorType::UniformTexelBuffer`]: crate::DescriptorType::UniformTexelBuffer
        const UNIFORM_TEXEL = 1 << 2;
        /// The buffer can be used to create a [`BufferView`] suitable for occupying a
        /// [`DescriptorSet`] slot of type [`DescriptorType::StorageTexelBuffer`].
        ///
        /// [`BufferView`]: crate::BufferView
        /// [`DescriptorSet`]: crate::DescriptorSet
        /// [`DescriptorType::StorageTexelBuffer`]: crate::DescriptorType::StorageTexelBuffer
        const STORAGE_TEXEL = 1 << 3;
        /// The buffer can be used to create a [`BufferView`] suitable for occupying a
        /// [`DescriptorSet`] slot of type [`DescriptorType::UniformBuffer`] or
        /// [`DescriptorType::UniformBufferDynamic`].
        ///
        /// [`BufferView`]: crate::BufferView
        /// [`DescriptorSet`]: crate::DescriptorSet
        /// [`DescriptorType::UniformBuffer`]: crate::DescriptorType::UniformBuffer
        /// [`DescriptorType::UniformBufferDynamic`]: crate::DescriptorType::UniformBufferDynamic
        const UNIFORM = 1 << 4;
        /// The buffer can be used to create a [`BufferView`] suitable for occupying a
        /// [`DescriptorSet`] slot of type [`DescriptorType::StorageBuffer`] or
        /// [`DescriptorType::StorageBufferDynamic`].
        ///
        /// [`BufferView`]: crate::BufferView
        /// [`DescriptorSet`]: crate::DescriptorSet
        /// [`DescriptorType::StorageBuffer`]: crate::DescriptorType::StorageBuffer
        /// [`DescriptorType::StorageBufferDynamic`]: crate::DescriptorType::StorageBufferDynamic
        const STORAGE = 1 << 5;
        /// The buffer can be used as the source of an index buffer bind command.
        const INDEX = 1 << 6;
        /// The buffer can be used as the source of a vertex buffer bind command.
        const VERTEX = 1 << 7;
        /// The buffer can be used as the source of a draw indirect command.
        const INDIRECT = 1 << 8;
        /// The buffer is suitable for use in conditional rendering commands.
        const CONDITIONAL_RENDERING = 1 << 9;
        /// The buffer can be used to retrieve a buffer device address and
        /// use that address to access the buffer's memory from a shader.
        const SHADER_DEVICE_ADDRESS = 1 << 17;
    }
}

impl FromGfx<BufferUsage> for vk::BufferUsageFlags {
    fn from_gfx(value: BufferUsage) -> Self {
        let mut flags = vk::BufferUsageFlags::empty();
        if value.contains(BufferUsage::TRANSFER_SRC) {
            flags |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if value.contains(BufferUsage::TRANSFER_DST) {
            flags |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if value.contains(BufferUsage::UNIFORM_TEXEL) {
            flags |= vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
        }
        if value.contains(BufferUsage::STORAGE_TEXEL) {
            flags |= vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
        }
        if value.contains(BufferUsage::UNIFORM) {
            flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if value.contains(BufferUsage::STORAGE) {
            flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if value.contains(BufferUsage::INDEX) {
            flags |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if value.contains(BufferUsage::VERTEX) {
            flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if value.contains(BufferUsage::INDIRECT) {
            flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
        if value.contains(BufferUsage::CONDITIONAL_RENDERING) {
            flags |= vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT;
        }
        if value.contains(BufferUsage::SHADER_DEVICE_ADDRESS) {
            flags |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
        flags
    }
}

bitflags::bitflags! {
    /// Bitmask specifying properties for a memory type.
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct MemoryUsage: u8 {
        /// Hints for allocator to find memory with faster device access.
        const FAST_DEVICE_ACCESS = 0x01;

        /// Hints allocator that memory will be used for data downloading.
        /// Allocator will strongly prefer host-cached memory.
        const DOWNLOAD = 0x04;

        /// Hints allocator that memory will be used for data uploading.
        /// If `DOWNLOAD` flag is not set then allocator will assume that
        /// host will access memory in write-only manner and may
        /// pick not host-cached.
        const UPLOAD = 0x08;

        /// Hints allocator that memory will be used for short duration
        /// allowing to use faster algorithm with less memory overhead.
        /// If use holds returned memory block for too long then
        /// effective memory overhead increases instead.
        /// Best use case is for staging buffer for single batch of operations.
        const TRANSIENT = 0x10;
    }
}

/// A wrapper around a Vulkan buffer object.
///
/// Buffers represent linear arrays of data which are used for various purposes
/// by binding them to a graphics or compute pipeline via descriptor sets or
/// certain commands, or by directly specifying them as parameters to certain commands.
#[derive(Clone)]
#[repr(transparent)]
pub struct Buffer {
    inner: Arc<Inner>,
}

impl Buffer {
    pub fn info(&self) -> &BufferInfo {
        &self.inner.info
    }

    pub fn address(&self) -> Option<DeviceAddress> {
        self.inner.address
    }

    pub fn handle(&self) -> vk::Buffer {
        self.inner.handle
    }

    pub fn try_into_mappable(mut self) -> Result<MappableBuffer, Self> {
        if self.is_mappable() && Arc::get_mut(&mut self.inner).is_some() {
            Ok(MappableBuffer { buffer: self })
        } else {
            Err(self)
        }
    }

    pub fn try_as_mappable(&mut self) -> Option<&mut MappableBuffer> {
        if self.is_mappable() && Arc::get_mut(&mut self.inner).is_some() {
            // SAFETY: buffer is unique and mappable
            Some(unsafe { MappableBuffer::wrap(self) })
        } else {
            None
        }
    }

    fn is_mappable(&self) -> bool {
        self.inner
            .memory_usage
            .intersects(gpu_alloc::UsageFlags::DOWNLOAD | gpu_alloc::UsageFlags::UPLOAD)
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            // SAFETY: unique access is guaranteed by the interface
            let memory_block = unsafe { &*self.inner.memory_block.get() };

            f.debug_struct("Buffer")
                .field("info", &self.inner.info)
                .field("owner", &self.inner.owner)
                .field("handle", &self.inner.handle)
                .field("address", &self.inner.address)
                .field("memory_handle", &self.inner.memory)
                .field("memory_offset", &memory_block.offset())
                .field("memory_size", &memory_block.size())
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for Buffer {}
impl PartialEq for Buffer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Hash for Buffer {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

impl From<MappableBuffer> for Buffer {
    #[inline]
    fn from(value: MappableBuffer) -> Self {
        value.buffer
    }
}

/// A host-visible buffer that can be mapped for reading and/or writing.
///
/// See [`Buffer`] for more information.
#[derive(Eq, PartialEq)]
#[repr(transparent)]
pub struct MappableBuffer {
    buffer: Buffer,
}

impl MappableBuffer {
    pub(crate) fn new(
        handle: vk::Buffer,
        info: BufferInfo,
        memory_usage: gpu_alloc::UsageFlags,
        address: Option<DeviceAddress>,
        owner: WeakDevice,
        memory_block: MemoryBlock<vk::DeviceMemory>,
    ) -> Self {
        Self {
            buffer: Buffer {
                inner: Arc::new(Inner {
                    handle,
                    info,
                    memory_usage,
                    address,
                    owner,
                    memory: *memory_block.memory(),
                    memory_block: UnsafeCell::new(ManuallyDrop::new(memory_block)),
                }),
            },
        }
    }

    unsafe fn wrap(buffer: &mut Buffer) -> &mut Self {
        &mut *(buffer as *mut Buffer).cast::<Self>()
    }

    pub fn info(&self) -> &BufferInfo {
        self.buffer.info()
    }

    pub fn address(&self) -> Option<DeviceAddress> {
        self.buffer.address()
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer.handle()
    }

    pub fn freeze(self) -> Buffer {
        self.buffer
    }

    /// # Safety
    ///
    /// The following must be true:
    /// - Returned mutable reference must not be used to replace the value.
    pub(crate) unsafe fn memory_block(&mut self) -> &mut MemoryBlock<vk::DeviceMemory> {
        &mut *self.buffer.inner.memory_block.get()
    }
}

struct Inner {
    handle: vk::Buffer,
    info: BufferInfo,
    memory_usage: gpu_alloc::UsageFlags,
    address: Option<DeviceAddress>,
    owner: WeakDevice,
    memory: vk::DeviceMemory,
    memory_block: UnsafeCell<ManuallyDrop<MemoryBlock<vk::DeviceMemory>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let block = ManuallyDrop::take(self.memory_block.get_mut());

            if let Some(device) = self.owner.upgrade() {
                device.destroy_buffer(self.handle, block);
            }

            // NOTE: `Relevant` will println error here if device was already destroyed
        }
    }
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
