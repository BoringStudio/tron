use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use anyhow::Result;

use crate::util::{BindlessResources, StorageBufferHandle};

// TODO: reuse buffers
#[derive(Default)]
pub struct MultiBufferArena {
    retired: Mutex<Vec<StorageBufferHandle>>,
}

impl MultiBufferArena {
    pub fn begin<T: gfx::Std430>(
        &self,
        device: &gfx::Device,
        capacity: usize,
        usage: gfx::BufferUsage,
    ) -> Result<BufferArena<T>> {
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
        let mut retired = self.retired.lock().unwrap();
        for target in retired.drain(..) {
            bindless_resources.free_storage_buffer(target);
        }
    }
}

pub struct BufferArena<T> {
    buffer: gfx::Buffer,
    ptr: *mut MaybeUninit<u8>,
    offset: usize,
    capacity: usize,
    _makrer: PhantomData<T>,
}

impl<T> BufferArena<T>
where
    T: gfx::Std430,
{
    const ITEM_SIZE: usize = gfx::align_size(T::ALIGN_MASK, std::mem::size_of::<T>());

    pub fn write(&mut self, data: &T) {
        assert!(self.offset + Self::ITEM_SIZE <= self.capacity);

        unsafe {
            std::ptr::copy_nonoverlapping(
                (data as *const T).cast(),
                self.ptr.add(self.offset),
                std::mem::size_of::<T>(),
            )
        }

        self.offset += Self::ITEM_SIZE;
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

unsafe impl<T> Send for BufferArena<T> {}
