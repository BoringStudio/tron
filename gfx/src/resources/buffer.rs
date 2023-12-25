use std::cell::UnsafeCell;
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use gpu_alloc::MemoryBlock;
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::types::DeviceAddress;

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

impl From<IndexType> for vk::IndexType {
    fn from(value: IndexType) -> Self {
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
    pub usage: vk::BufferUsageFlags,
}

#[derive(Clone)]
pub struct Buffer {
    info: BufferInfo,
    memory_usage: gpu_alloc::UsageFlags,
    address: Option<DeviceAddress>,
    inner: Arc<Inner>,
}

impl Buffer {
    pub fn info(&self) -> &BufferInfo {
        &self.info
    }

    pub fn address(&self) -> Option<DeviceAddress> {
        self.address
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
        self.memory_usage
            .intersects(gpu_alloc::UsageFlags::DOWNLOAD | gpu_alloc::UsageFlags::UPLOAD)
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            // SAFETY: unique access is guaranteed by inderface
            let memory_block = unsafe { &*self.inner.memory_block.get() };

            f.debug_struct("Buffer")
                .field("info", &self.info)
                .field("owner", &self.inner.owner)
                .field("handle", &self.inner.handle)
                .field("address", &self.address)
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
    pub fn new(
        handle: vk::Buffer,
        info: BufferInfo,
        memory_usage: gpu_alloc::UsageFlags,
        address: Option<DeviceAddress>,
        owner: WeakDevice,
        memory_block: MemoryBlock<vk::DeviceMemory>,
    ) -> Self {
        Self {
            buffer: Buffer {
                info,
                memory_usage,
                address,
                inner: Arc::new(Inner {
                    handle,
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
