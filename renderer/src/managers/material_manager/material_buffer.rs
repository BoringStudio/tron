use anyhow::Result;

use crate::util::{ScatterCopy, ScatterData};

pub struct MaterialBuffer {
    buffer: gfx::Buffer,
    current_count: usize,
    reserved_count: usize,
    updated_slots: Vec<usize>,
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
                buffer: make_buffer(
                    device,
                    align_mask,
                    (item_size * MaterialBuffer::INITIAL_CAPACITY) as u64,
                )?,
                current_count: MaterialBuffer::INITIAL_CAPACITY,
                reserved_count: MaterialBuffer::INITIAL_CAPACITY,
                updated_slots: Vec::new(),
            })
        }

        new_impl(device, T::ALIGN_MASK, std::mem::size_of::<T>())
    }

    pub fn update_slot(&mut self, slot: usize) {
        if slot > self.reserved_count {
            self.reserved_count = slot.next_power_of_two();
        }
        self.updated_slots.push(slot);
    }

    /// # Safety
    /// - `T` must be the same type as the one used to construct the buffer.
    pub unsafe fn flush<T, F>(
        &mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        scatter_copy: &ScatterCopy,
        mut get_data: F,
    ) -> Result<()>
    where
        T: gfx::Std430,
        F: FnMut(usize) -> T,
    {
        let item_size = std::mem::size_of::<T>();

        // NOTE: `reserved_count` is eventially updated on `update_index` calls.
        if self.reserved_count != self.current_count {
            let buffer = make_buffer(
                device,
                T::ALIGN_MASK,
                (item_size * self.reserved_count) as u64,
            )?;
            let old_buffer = std::mem::replace(&mut self.buffer, buffer);

            encoder.copy_buffer(
                &old_buffer,
                &self.buffer,
                &[gfx::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: (item_size * self.current_count) as u64,
                }],
            );

            self.current_count = self.reserved_count;
        }

        if self.updated_slots.is_empty() {
            return Ok(());
        }

        let data = self
            .updated_slots
            .drain(..)
            .map(|slot| ScatterData::new((item_size * slot) as u32, get_data(slot)));

        scatter_copy.execute(device, encoder, &self.buffer, data)?;
        Ok(())
    }
}

impl std::ops::Deref for MaterialBuffer {
    type Target = gfx::Buffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer
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
