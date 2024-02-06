use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use anyhow::Result;
use shared::FastHashMap;

use crate::util::{BindlessResources, StorageBufferHandle};

#[derive(Default)]
pub struct MultiBufferArena {
    buffer_align_mask: usize,
    buffers: Mutex<FastHashMap<gfx::BufferUsage, Buffers>>,
}

impl MultiBufferArena {
    pub fn new(device: &gfx::Device) -> Self {
        let buffer_align_mask = device.limits().min_storage_buffer_offset_alignment as usize - 1;
        Self {
            buffer_align_mask,
            buffers: Mutex::new(FastHashMap::default()),
        }
    }

    pub fn begin<T: gfx::Std430>(
        &self,
        device: &gfx::Device,
        capacity: usize,
        usage: gfx::BufferUsage,
    ) -> Result<BufferArena<T>> {
        let mapped_buffer = 'existing: {
            let mut buffers = self.buffers.lock().unwrap();
            let Some(buffers) = buffers.get_mut();
            if let Some(buffers) = buffers.get(&usage) {

            }
        };

        let size = capacity * BufferArena::<T>::ITEM_SIZE;
        let buffer = device.create_mappable_buffer(
            gfx::BufferInfo {
                align_mask: T::ALIGN_MASK,
                size,
                usage,
            },
            gfx::MemoryUsage::UPLOAD,
        )?;
        // TODO: is TRANSIENT needed here?

        let ptr = device
            .map_memory(&mut buffer.as_mappable(), 0, size)?
            .as_mut_ptr();

        Ok(BufferArena {
            buffer,
            ptr,
            offset: 0,
            capacity: size,
            _makrer: PhantomData,
        })
    }

    pub fn end<T: gfx::Std430>(
        &self,
        device: &gfx::Device,
        bindless_resources: &BindlessResources,
        arena: BufferArena<T>,
    ) -> StorageBufferHandle {
        device.unmap_memory(&mut arena.buffer.as_mappable());
        let handle = bindless_resources.alloc_storage_buffer(device, arena.buffer);
        self.retired.lock().unwrap().push(handle);
        handle
    }

    pub fn flush(&self, bindless_resources: &BindlessResources) {
        let mut groups = self.buffers.lock().unwrap();
        for buffers in groups.values_mut() {
            for mut buffer in buffers.retired.drain(..) {
                bindless_resources.free_storage_buffers_batch(&buffer.handles);
                buffer.offset = 0;
                buffer.handles.clear();
                buffers.free.push(buffer);
            }

            buffers.retired.append(&mut buffers.used);
        }
    }
}

struct Buffers {
    used: Vec<MappedBuffer>,
    free: Vec<MappedBuffer>,
    retired: Vec<MappedBuffer>,
}

struct MappedBuffer {
    buffer: gfx::Buffer,
    ptr: *mut MaybeUninit<u8>,
    offset: usize,
    capacity: usize,
    handles: Vec<StorageBufferHandle>,
}

unsafe impl Send for MappedBuffer {}

pub struct BufferArena<T> {
    inner: MappedBuffer,
    _makrer: PhantomData<T>,
}

impl<T> BufferArena<T>
where
    T: gfx::Std430,
{
    const ITEM_SIZE: usize = gfx::align_size(T::ALIGN_MASK, std::mem::size_of::<T>());

    pub fn write(&mut self, data: &T) {
        assert!(self.inner.offset + Self::ITEM_SIZE <= self.inner.capacity);

        unsafe {
            std::ptr::copy_nonoverlapping(
                (data as *const T).cast(),
                self.inner.ptr.add(self.inner.offset),
                std::mem::size_of::<T>(),
            )
        }

        self.inner.offset += Self::ITEM_SIZE;
    }
}
