use anyhow::Result;

use crate::util::{ScatterCopy, ScatterData};

pub struct MaterialBuffer {
    targets: [Target; 2],
    current_target: bool,
    reserved_count: usize,
}

impl MaterialBuffer {
    const INITIAL_CAPACITY: usize = 16;

    pub fn new<T: gfx::Std430>(device: &gfx::Device) -> Result<Self, gfx::OutOfDeviceMemory> {
        fn new_impl(
            device: &gfx::Device,
            align_mask: u64,
            item_size: usize,
        ) -> Result<MaterialBuffer, gfx::OutOfDeviceMemory> {
            Ok(MaterialBuffer {
                targets: [
                    Target::new(
                        device,
                        align_mask,
                        item_size,
                        MaterialBuffer::INITIAL_CAPACITY,
                    )?,
                    Target::new(
                        device,
                        align_mask,
                        item_size,
                        MaterialBuffer::INITIAL_CAPACITY,
                    )?,
                ],
                current_target: false,
                reserved_count: MaterialBuffer::INITIAL_CAPACITY,
            })
        }

        new_impl(device, T::ALIGN_MASK, std::mem::size_of::<T>())
    }

    pub fn update_slot(&mut self, slot: usize) {
        let target = &mut self.targets[self.current_target as usize];

        if slot > self.reserved_count {
            self.reserved_count = slot.next_power_of_two();
        }
        target.updated_slots.insert(slot);
    }

    /// # Safety
    /// - `T` must be the same type as the one used to construct the buffer.
    pub unsafe fn flush<'a, T, F>(
        &'a mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        scatter_copy: &ScatterCopy,
        mut get_data: F,
    ) -> Result<&'a gfx::Buffer>
    where
        T: gfx::Std430,
        F: FnMut(usize) -> T,
    {
        let item_size = std::mem::size_of::<T>();

        let (current_target, prev_target) = {
            let [front, back] = &mut self.targets;
            if self.current_target {
                (back, front)
            } else {
                (front, back)
            }
        };

        // NOTE: `reserved_count` is eventially updated on `update_index` calls.
        if self.reserved_count != current_target.current_count {
            let buffer = make_buffer(
                device,
                T::ALIGN_MASK,
                (item_size * self.reserved_count) as u64,
            )?;
            let old_buffer = std::mem::replace(&mut current_target.buffer, buffer);

            encoder.copy_buffer(
                &old_buffer,
                &current_target.buffer,
                &[gfx::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (item_size * current_target.current_count) as u64,
                }],
            );

            current_target.current_count = self.reserved_count;
        }

        if current_target.updated_slots.is_empty() && prev_target.updated_slots.is_empty() {
            return Ok(&current_target.buffer);
        }

        let data = current_target
            .updated_slots
            .merge_iter(&prev_target.updated_slots)
            .map(|slot| ScatterData::new((item_size * slot) as u32, get_data(slot)));

        scatter_copy.execute(device, encoder, &current_target.buffer, data)?;

        // Clear previous target updated slots as they are no longer needed.
        prev_target.updated_slots.clear();

        self.current_target = !self.current_target;
        Ok(&current_target.buffer)
    }
}

struct Target {
    buffer: gfx::Buffer,
    current_count: usize,
    updated_slots: UpdatedSlots,
}

impl Target {
    fn new(
        device: &gfx::Device,
        align_mask: u64,
        item_size: usize,
        capacity: usize,
    ) -> Result<Self, gfx::OutOfDeviceMemory> {
        Ok(Target {
            buffer: make_buffer(device, align_mask, (item_size * capacity) as u64)?,
            current_count: capacity,
            updated_slots: Default::default(),
        })
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
    fn insert(&mut self, slot: usize) {
        let chunk = slot / BITS_PER_CHUNK;
        let bit = slot % BITS_PER_CHUNK;

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
    ) -> impl Iterator<Item = usize> + ExactSizeIterator + 'a {
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
                    offset: i * BITS_PER_CHUNK,
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
    I: Iterator<Item = usize>,
{
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total, Some(self.total))
    }
}

impl<I> ExactSizeIterator for ChunksIter<I> where I: Iterator<Item = usize> {}

struct ChunkIter {
    chunk: SlotChunk,
    offset: usize,
}

impl Iterator for ChunkIter {
    type Item = usize;

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

        Some(self.offset + mask.trailing_zeros() as usize)
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
