use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use anyhow::Result;
use shared::FastHashMap;

use crate::util::{BindlessResources, StorageBufferHandle};

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
        fn begin_impl(
            this: &MultiBufferArena,
            device: &gfx::Device,
            align_mask: usize,
            size: usize,
            usage: gfx::BufferUsage,
        ) -> Result<MappedBuffer> {
            // Find an existing buffer
            if let Some(buffers) = this.buffers.lock().unwrap().get_mut(&usage) {
                for (i, buffer) in buffers.used.iter().enumerate() {
                    if buffer.capacity >= gfx::align_offset(align_mask, buffer.offset) + size {
                        let mut buffer = buffers.used.swap_remove(i);
                        buffer.offset = gfx::align_offset(align_mask, buffer.offset);
                        return Ok(buffer);
                    }
                }
                for (i, buffer) in buffers.free.iter().enumerate() {
                    if buffer.capacity >= size {
                        let buffer = buffers.free.swap_remove(i);
                        debug_assert_eq!(buffer.offset, 0);
                        return Ok(buffer);
                    }
                }
            }

            // Create new buffer
            let capacity = size.next_power_of_two();
            let buffer = device.create_mappable_buffer(
                gfx::BufferInfo {
                    align_mask,
                    size: capacity,
                    usage,
                },
                gfx::MemoryUsage::UPLOAD,
            )?;

            let ptr = device
                .map_memory(&mut buffer.as_mappable(), 0, capacity)?
                .as_mut_ptr();

            Ok(MappedBuffer {
                buffer,
                ptr,
                offset: 0,
                capacity,
                handles: Vec::new(),
            })
        }

        let size = capacity * BufferArena::<T>::ITEM_SIZE;
        let mapped = begin_impl(self, device, T::ALIGN_MASK, size, usage)?;
        Ok(BufferArena {
            initial_offset: mapped.offset,
            inner: mapped,
            size,
            _makrer: PhantomData,
        })
    }

    pub fn end_raw<T: gfx::Std430>(&self, arena: BufferArena<T>) -> gfx::BufferRange {
        let BufferArena {
            inner: mut mapped,
            initial_offset,
            size,
            ..
        } = arena;
        mapped.offset = gfx::align_offset(T::ALIGN_MASK | self.buffer_align_mask, mapped.offset);

        let usage = mapped.buffer.info().usage;
        let range = gfx::BufferRange {
            buffer: mapped.buffer.clone(),
            offset: initial_offset,
            size,
        };

        let mut buffers = self.buffers.lock().unwrap();
        buffers.entry(usage).or_default().used.push(mapped);
        range
    }

    pub fn end<T: gfx::Std430>(
        &self,
        device: &gfx::Device,
        bindless_resources: &BindlessResources,
        arena: BufferArena<T>,
    ) -> StorageBufferHandle {
        fn end_impl(
            this: &MultiBufferArena,
            device: &gfx::Device,
            bindless_resources: &BindlessResources,
            mut mapped: MappedBuffer,
            initial_offset: usize,
            size: usize,
        ) -> StorageBufferHandle {
            let usage = mapped.buffer.info().usage;
            let handle = bindless_resources.alloc_storage_buffer(
                device,
                gfx::BufferRange {
                    buffer: mapped.buffer.clone(),
                    offset: initial_offset,
                    size,
                },
            );
            mapped.handles.push(handle);

            let mut buffers = this.buffers.lock().unwrap();
            buffers.entry(usage).or_default().used.push(mapped);
            handle
        }

        let BufferArena {
            inner: mut mapped,
            initial_offset,
            size,
            ..
        } = arena;
        mapped.offset = gfx::align_offset(T::ALIGN_MASK | self.buffer_align_mask, mapped.offset);

        end_impl(
            self,
            device,
            bindless_resources,
            mapped,
            initial_offset,
            size,
        )
    }

    pub fn flush(&self, bindless_resources: &BindlessResources) {
        let mut groups = self.buffers.lock().unwrap();
        for buffers in groups.values_mut() {
            for mut buffer in buffers.retired.drain(..) {
                if !buffer.handles.is_empty() {
                    bindless_resources.free_storage_buffers_batch(&buffer.handles);
                }

                buffer.offset = 0;
                buffer.handles.clear();
                buffers.free.push(buffer);
            }

            buffers.retired.append(&mut buffers.used);
        }
    }
}

#[derive(Default)]
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

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        if let Some(device) = self.buffer.owner().upgrade() {
            device.unmap_memory(&mut self.buffer.as_mappable());
        }
    }
}

pub struct BufferArena<T> {
    inner: MappedBuffer,
    initial_offset: usize,
    size: usize,
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

    pub fn as_mut_ptr(&mut self) -> *mut MaybeUninit<u8> {
        assert!(self.inner.offset <= self.inner.capacity);
        unsafe { self.inner.ptr.add(self.inner.offset) }
    }

    /// # Safety
    /// The following must be true:
    /// - `offset` must be a multiple of `T::ALIGN_MASK`
    /// - The result `offset` must be less than or equal to `self.size`
    pub unsafe fn add_offset(&mut self, offset: usize) {
        self.inner.offset += offset;
    }
}
