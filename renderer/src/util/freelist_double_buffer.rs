use anyhow::Result;

use crate::util::{ScatterCopy, ScatterData};

pub struct FreelistDoubleBuffer {
    targets: [Target; 2],
    odd_target: bool,
    reserved_count: u32,
}

impl FreelistDoubleBuffer {
    pub fn with_capacity(initial_capacity: u32) -> Self {
        FreelistDoubleBuffer {
            targets: Default::default(),
            odd_target: false,
            reserved_count: initial_capacity,
        }
    }

    pub fn update_slot(&mut self, slot: u32) {
        let target = &mut self.targets[self.odd_target as usize];

        if slot > self.reserved_count {
            self.reserved_count = slot.checked_next_power_of_two().expect("too many slots");
        }
        target.updated_slots.insert(slot);
    }

    /// # Safety
    /// - `T` must be the same type on each invocation.
    #[inline]
    pub unsafe fn flush<'a, T, F>(
        &'a mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        scatter_copy: &ScatterCopy,
        mut get_data: F,
    ) -> Result<&'a gfx::Buffer>
    where
        T: gfx::Std430,
        F: FnMut(u32) -> T,
    {
        let mut item_size = std::mem::size_of::<T>();
        if item_size & T::ALIGN_MASK as usize != 0 {
            item_size += T::ALIGN_MASK as usize + 1 - (item_size & T::ALIGN_MASK as usize);
        }

        let (current_target, prev_target) = {
            let [front, back] = &mut self.targets;
            if self.odd_target {
                (back, front)
            } else {
                (front, back)
            }
        };

        // NOTE: `reserved_count` is eventually updated on `update_index` calls.
        let (current_target_buffer, updated_slots) = current_target.prepare(
            device,
            encoder,
            self.reserved_count,
            item_size,
            T::ALIGN_MASK,
        )?;

        if updated_slots.is_empty() && prev_target.updated_slots.is_empty() {
            return Ok(current_target_buffer);
        }

        let data = updated_slots
            .merge_iter(&prev_target.updated_slots)
            .map(|slot| ScatterData::new(item_size as u32 * slot, get_data(slot)));

        scatter_copy.execute(device, encoder, current_target_buffer, data)?;

        // Clear previous target updated slots as they are no longer needed.
        prev_target.updated_slots.clear();

        self.odd_target = !self.odd_target;
        Ok(current_target_buffer)
    }
}

#[derive(Default)]
struct Target {
    buffer: Option<gfx::Buffer>,
    current_count: u32,
    updated_slots: UpdatedSlots,
}

impl Target {
    fn prepare<'a>(
        &'a mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        reserved_count: u32,
        item_size: usize,
        align_mask: u64,
    ) -> Result<(&'a mut gfx::Buffer, &'a mut UpdatedSlots), gfx::OutOfDeviceMemory> {
        if self.buffer.is_some() && self.current_count == reserved_count {
            // SAFETY: `self.buffer` is `Some`
            // NOTE: borrow checker is mad, I am too!
            let buffer = unsafe { self.buffer.as_mut().unwrap_unchecked() };
            return Ok((buffer, &mut self.updated_slots));
        }

        let old_buffer = self.buffer.take();
        let buffer = self.buffer.get_or_insert(make_buffer(
            device,
            align_mask,
            item_size as u64 * reserved_count as u64,
        )?);

        if let Some(old_buffer) = old_buffer {
            encoder.copy_buffer(
                &old_buffer,
                buffer,
                &[gfx::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: item_size as u64 * self.current_count as u64,
                }],
            );
        }

        self.current_count = reserved_count;
        Ok((buffer, &mut self.updated_slots))
    }
}

fn make_buffer(
    device: &gfx::Device,
    align_mask: u64,
    size: u64,
) -> Result<gfx::Buffer, gfx::OutOfDeviceMemory> {
    device.create_buffer(gfx::BufferInfo {
        align: align_mask | MIN_ALIGN_MASK,
        size,
        usage: gfx::BufferUsage::STORAGE
            | gfx::BufferUsage::TRANSFER_DST
            | gfx::BufferUsage::TRANSFER_SRC,
    })
}

const MIN_ALIGN_MASK: u64 = 0b1111;

struct UpdatedSlots {
    chunks: Vec<SlotChunk>,
    is_empty: bool,
}

impl Default for UpdatedSlots {
    fn default() -> Self {
        UpdatedSlots {
            chunks: Vec::new(),
            is_empty: true,
        }
    }
}

impl UpdatedSlots {
    fn insert(&mut self, slot: u32) {
        let chunk = (slot as usize) / BITS_PER_CHUNK;
        let bit = (slot as usize) % BITS_PER_CHUNK;

        if chunk >= self.chunks.len() {
            self.chunks.resize(chunk + 1, 0);
        }
        self.chunks[chunk] |= 1 << bit;
        self.is_empty = false;
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.is_empty = true;
    }

    fn is_empty(&self) -> bool {
        self.is_empty
    }

    fn merge_iter<'a>(
        &'a self,
        prev: &'a UpdatedSlots,
    ) -> impl Iterator<Item = u32> + ExactSizeIterator + 'a {
        let cur_len = self.chunks.len();
        let prev_len = prev.chunks.len();

        let (cur, prev, rest) = if cur_len < prev_len {
            let (prev, rest) = prev.chunks.split_at(cur_len);
            (self.chunks.as_slice(), prev, rest)
        } else {
            let (cur, rest) = self.chunks.split_at(prev_len);
            (cur, prev.chunks.as_slice(), rest)
        };

        let total = std::iter::zip(cur, prev)
            .map(|(cur, prev)| cur | prev)
            .chain(rest.iter().copied())
            .map(|chunk| chunk.count_ones() as usize)
            .sum::<usize>();

        ChunksIter {
            inner: std::iter::zip(cur, prev)
                .map(|(cur, prev)| cur | prev)
                .chain(rest.iter().copied())
                .enumerate()
                .flat_map(|(i, chunk)| ChunkIter {
                    chunk,
                    offset: (i * BITS_PER_CHUNK) as u32,
                }),
            total,
        }
    }
}

struct ChunksIter<I> {
    inner: I,
    total: usize,
}

impl<I> Iterator for ChunksIter<I>
where
    I: Iterator<Item = u32>,
{
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total, Some(self.total))
    }
}

impl<I> ExactSizeIterator for ChunksIter<I> where I: Iterator<Item = u32> {}

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
