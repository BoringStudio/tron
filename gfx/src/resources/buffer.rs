use std::cell::UnsafeCell;
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use gpu_alloc::MemoryBlock;
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::types::DeviceAddress;
use crate::util::FromGfx;

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
}

impl FromGfx<IndexType> for vk::IndexType {
    fn from_gfx(value: IndexType) -> Self {
        match value {
            IndexType::U16 => vk::IndexType::UINT16,
            IndexType::U32 => vk::IndexType::UINT32,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferInfo {
    pub align: u64,
    pub size: u64,
    pub usage: BufferUsage,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SRC = 1;
        const TRANSFER_DST = 1 << 1;
        const UNIFORM_TEXEL = 1 << 2;
        const STORAGE_TEXEL = 1 << 3;
        const UNIFORM = 1 << 4;
        const STORAGE = 1 << 5;
        const INDEX = 1 << 6;
        const VERTEX = 1 << 7;
        const INDIRECT = 1 << 8;
        const CONDITIONAL_RENDERING = 1 << 9;

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

#[derive(Eq, PartialEq)]
#[repr(transparent)]
pub struct MappableBuffer {
    buffer: Buffer,
}

impl std::ops::Deref for MappableBuffer {
    type Target = Buffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
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

    pub fn freeze(self) -> Buffer {
        self.buffer
    }

    /// # Safety
    ///
    /// The following must be true:
    /// - Returned mutable reference must not be used to replace the value.
    pub unsafe fn memory_block(&mut self) -> &mut MemoryBlock<vk::DeviceMemory> {
        &mut *self.inner.memory_block.get()
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
