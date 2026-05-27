use anyhow::Result;

use crate::util::{
    BindlessResources, MultiBufferArena, ScatterCopy, ScatterData, StorageBufferHandle,
};

pub struct FreelistDoubleBuffer {
    targets: [Target; 2],
    handle: StorageBufferHandle,
    odd_target: bool,
    reserved_count: u32,
}

impl FreelistDoubleBuffer {
    pub fn with_capacity(initial_capacity: u32) -> Self {
        FreelistDoubleBuffer {
            targets: Default::default(),
            handle: StorageBufferHandle::INVALID,
            odd_target: false,
            reserved_count: initial_capacity,
        }
    }

    #[allow(dead_code)]
    pub fn handle(&self) -> StorageBufferHandle {
        self.handle
    }

    pub fn update_slot(&mut self, slot: u32) {
        let target = &mut self.targets[self.odd_target as usize];

        if slot > self.reserved_count {
            self.reserved_count = slot.checked_next_power_of_two().expect("too many slots");
        }
        target.updated_slots.insert(slot);
    }

    pub fn remove_slot(&mut self, slot: u32) {
        self.targets.iter_mut().for_each(|item| {
            item.updated_slots.remove(slot);
        });
    }

    /// # Safety
    /// - `T` must be the same type on each invocation.
    #[inline]
    pub unsafe fn flush<T, F>(
        &mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        scatter_copy: &ScatterCopy,
        bindless_resources: &BindlessResources,
        buffers: &MultiBufferArena,
        mut get_data: F,
    ) -> Result<()>
    where
        T: gfx::Std430,
        F: FnMut(u32) -> T,
    {
        let item_size = gfx::align_size(T::ALIGN_MASK, std::mem::size_of::<T>());

        let (current_target, prev_target) = {
            let [front, back] = &mut self.targets;
            if self.odd_target {
                (back, front)
            } else {
                (front, back)
            }
        };

        // NOTE: `reserved_count` is eventually updated on `update_index` calls.
        let prepared = current_target.prepare(
            device,
            encoder,
            bindless_resources,
            self.reserved_count,
            item_size,
            T::ALIGN_MASK,
        )?;
        self.handle = prepared.handle;

        if prepared.updated_slots.is_empty() && prev_target.updated_slots.is_empty() {
            return Ok(());
        }

        let (count, data) = prepared
            .updated_slots
            .merge_iter(&prev_target.updated_slots);
        let data = data.map(|slot| ScatterData::new(item_size as u32 * slot, get_data(slot)));

        scatter_copy.execute(device, encoder, prepared.buffer, buffers, count, data)?;

        // Clear previous target updated slots as they are no longer needed.
        prev_target.updated_slots.clear();

        self.odd_target = !self.odd_target;
        Ok(())
    }
}

#[derive(Default)]
struct Target {
    buffer: Option<(gfx::Buffer, StorageBufferHandle)>,
    current_count: u32,
    updated_slots: UpdatedSlots,
}

impl Target {
    fn prepare<'a>(
        &'a mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        bindless_resources: &BindlessResources,
        reserved_count: u32,
        item_size: usize,
        align_mask: usize,
    ) -> Result<PreparedTarget<'a>, gfx::OutOfDeviceMemory> {
        if self.buffer.is_some() && self.current_count == reserved_count {
            // SAFETY: `self.buffer` is `Some`
            // NOTE: borrow checker is mad, I am too!
            let (buffer, handle) = unsafe { self.buffer.as_ref().unwrap_unchecked() };
            return Ok(PreparedTarget {
                buffer,
                handle: *handle,
                updated_slots: &self.updated_slots,
            });
        }

        let old_buffer = self.buffer.take();
        let (buffer, handle) = {
            let buffer = make_buffer(device, align_mask, item_size * reserved_count as usize)?;
            let handle = bindless_resources
                .alloc_storage_buffer(device, gfx::BufferRange::whole(buffer.clone()));
            self.buffer.get_or_insert((buffer, handle))
        };

        if let Some((old_buffer, old_buffer_handle)) = old_buffer {
            bindless_resources.free_storage_buffer(old_buffer_handle);
            encoder.copy_buffer(
                &old_buffer,
                buffer,
                &[gfx::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: item_size * self.current_count as usize,
                }],
            );
        }

        self.current_count = reserved_count;
        Ok(PreparedTarget {
            buffer,
            handle: *handle,
            updated_slots: &self.updated_slots,
        })
    }
}

struct PreparedTarget<'a> {
    buffer: &'a gfx::Buffer,
    handle: StorageBufferHandle,
    updated_slots: &'a UpdatedSlots,
}

fn make_buffer(
    device: &gfx::Device,
    align_mask: usize,
    size: usize,
) -> Result<gfx::Buffer, gfx::OutOfDeviceMemory> {
    device.create_buffer(gfx::BufferInfo {
        align_mask: align_mask | MIN_ALIGN_MASK,
        size,
        usage: gfx::BufferUsage::STORAGE
            | gfx::BufferUsage::TRANSFER_DST
            | gfx::BufferUsage::TRANSFER_SRC,
    })
}

const MIN_ALIGN_MASK: usize = 0b1111;

struct UpdatedSlots {
    chunks: Vec<SlotChunk>,
    len: usize,
}

impl Default for UpdatedSlots {
    fn default() -> Self {
        UpdatedSlots {
            chunks: Vec::new(),
            len: 0,
        }
    }
}

impl UpdatedSlots {
    pub fn insert(&mut self, slot: u32) {
        let chunk = (slot as usize) / BITS_PER_CHUNK;
        let bit = (slot as usize) % BITS_PER_CHUNK;

        if chunk >= self.chunks.len() {
            self.chunks.resize(chunk + 1, 0);
        }
        let mask = 1 << bit;
        let chunk = &mut self.chunks[chunk];
        let chunk_before = *chunk;
        *chunk |= mask;
        self.len += (chunk_before & mask == 0) as usize;
    }

    pub fn remove(&mut self, slot: u32) {
        let chunk = (slot as usize) / BITS_PER_CHUNK;
        let bit = (slot as usize) % BITS_PER_CHUNK;

        if chunk >= self.chunks.len() {
            return;
        }
        let mask = 1 << bit;
        let chunk = &mut self.chunks[chunk];
        let chunk_before = *chunk;
        *chunk &= !mask;
        self.len -= (chunk_before & mask != 0) as usize;
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.len = 0;
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn merge_iter<'a>(&'a self, prev: &'a UpdatedSlots) -> (usize, impl Iterator<Item = u32> + 'a) {
        let chunks = MergedChunksIter {
            chunk: 0,
            left: &self.chunks,
            right: &prev.chunks,
        };

        let total = chunks
            .clone()
            .map(|chunk| chunk.count_ones() as usize)
            .sum::<usize>();

        let iter = chunks.enumerate().flat_map(|(i, chunk)| ChunkIter {
            chunk,
            offset: (i * BITS_PER_CHUNK) as u32,
        });
        (total, iter)
    }
}

#[derive(Clone)]
struct MergedChunksIter<'a> {
    chunk: usize,
    left: &'a [SlotChunk],
    right: &'a [SlotChunk],
}

impl Iterator for MergedChunksIter<'_> {
    type Item = SlotChunk;

    fn next(&mut self) -> Option<Self::Item> {
        let res = match (self.left.get(self.chunk), self.right.get(self.chunk)) {
            (None, None) => return None,
            (Some(left), None) => *left,
            (None, Some(right)) => *right,
            (Some(left), Some(right)) => *left | *right,
        };
        self.chunk += 1;
        Some(res)
    }
}

struct ChunkIter {
    chunk: SlotChunk,
    offset: u32,
}

impl Iterator for ChunkIter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk == 0 {
            return None;
        }

        //  10100 - 1     = 10011
        // !10011         = 01100
        //  10100 & 01100 = 00100
        let mask = self.chunk & !(self.chunk - 1);

        // 10100 & !00100 -> 10000
        self.chunk &= !mask;

        Some(self.offset + mask.trailing_zeros())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.chunk.count_ones() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for ChunkIter {}

type SlotChunk = u64;

const BITS_PER_CHUNK: usize = std::mem::size_of::<SlotChunk>() * 8;
