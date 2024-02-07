use std::mem::MaybeUninit;

use anyhow::Result;

use crate::util::{MultiBufferArena, ShaderPreprocessor};

pub struct ScatterData<T> {
    pub word_offset: u32,
    pub data: T,
}

impl<T> ScatterData<T> {
    pub fn new(byte_offset: u32, data: T) -> Self {
        Self {
            word_offset: byte_offset / 4,
            data,
        }
    }
}

pub struct ScatterCopy {
    descriptor_set_layout: gfx::DescriptorSetLayout,
    pipeline: gfx::ComputePipeline,
}

impl ScatterCopy {
    #[tracing::instrument(level = "debug", name = "create_scatter_copy", skip_all)]
    pub fn new(device: &gfx::Device, shader_preprocessor: &ShaderPreprocessor) -> Result<Self> {
        let shader = shader_preprocessor.begin().make_compute_shader(
            device,
            "/scatter_copy.comp",
            "main",
        )?;

        let descriptor_set_layout =
            device.create_descriptor_set_layout(gfx::DescriptorSetLayoutInfo {
                bindings: vec![
                    gfx::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: gfx::DescriptorType::StorageBuffer,
                        count: 1,
                        stages: gfx::ShaderStageFlags::COMPUTE,
                        flags: Default::default(),
                    },
                    gfx::DescriptorSetLayoutBinding {
                        binding: 1,
                        ty: gfx::DescriptorType::StorageBuffer,
                        count: 1,
                        stages: gfx::ShaderStageFlags::COMPUTE,
                        flags: Default::default(),
                    },
                ],
                flags: Default::default(),
            })?;

        let layout = device.create_pipeline_layout(gfx::PipelineLayoutInfo {
            sets: vec![descriptor_set_layout.clone()],
            push_constants: Vec::new(),
        })?;

        let pipeline =
            device.create_compute_pipeline(gfx::ComputePipelineInfo { shader, layout })?;

        Ok(Self {
            descriptor_set_layout,
            pipeline,
        })
    }

    pub fn execute<T, D>(
        &self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        dst: &gfx::Buffer,
        buffers: &MultiBufferArena,
        data: D,
    ) -> Result<()>
    where
        T: gfx::Std430,
        D: IntoIterator<Item = ScatterData<T>>,
        D::IntoIter: ExactSizeIterator,
    {
        let data = data.into_iter();

        let item_size = std::mem::size_of::<T>();
        assert_eq!(item_size % 4, 0);

        let count = data.len();
        let stride_bytes = item_size + 4;

        let buffer_size = 8 + count * stride_bytes;

        let staging_buffer = {
            let mut staging_buffer = buffers.begin::<u32>(
                device,
                buffer_size / 4,
                gfx::BufferUsage::STORAGE | gfx::BufferUsage::TRANSFER_SRC,
            )?;

            let ptr = staging_buffer.as_mut_ptr();
            debug_assert_eq!(ptr.align_offset(std::mem::align_of::<u32>()), 0);

            let mut writer = Writer { ptr, offset: 0 };

            unsafe {
                // words_to_copy
                writer.write_u32((item_size / 4) as u32);
                // count
                writer.write_u32(count as u32);
            }

            for item in data {
                unsafe {
                    writer.write_u32(item.word_offset);
                    writer.write_data(&item.data);
                }
            }

            unsafe { staging_buffer.add_offset(buffer_size) };

            buffers.end_raw(staging_buffer)
        };

        let descriptor_set = device.create_descriptor_set(gfx::DescriptorSetInfo {
            layout: self.descriptor_set_layout.clone(),
        })?;
        device.update_descriptor_sets(&[gfx::UpdateDescriptorSet {
            set: &descriptor_set,
            writes: &[
                gfx::DescriptorSetWrite {
                    binding: 0,
                    element: 0,
                    data: gfx::DescriptorSlice::StorageBuffer(&[staging_buffer]),
                },
                gfx::DescriptorSetWrite {
                    binding: 1,
                    element: 0,
                    data: gfx::DescriptorSlice::StorageBuffer(&[gfx::BufferRange::whole(
                        dst.clone(),
                    )]),
                },
            ],
        }]);

        encoder.bind_compute_pipeline(&self.pipeline);
        encoder.bind_compute_descriptor_sets(
            &self.pipeline.info().layout,
            0,
            &[&descriptor_set],
            &[],
        );

        encoder.memory_barrier(
            gfx::PipelineStageFlags::TRANSFER,
            gfx::AccessFlags::TRANSFER_WRITE,
            gfx::PipelineStageFlags::COMPUTE_SHADER,
            gfx::AccessFlags::SHADER_READ,
        );
        encoder.dispatch(((count + 63) / 64) as u32, 1, 1);

        Ok(())
    }
}

struct Writer {
    ptr: *mut MaybeUninit<u8>,
    offset: usize,
}

impl Writer {
    unsafe fn write_u32(&mut self, value: u32) {
        let value = value.to_le_bytes();
        std::ptr::copy_nonoverlapping(value.as_ptr().cast(), self.ptr.add(self.offset), 4);
        self.offset += 4;
    }

    unsafe fn write_data<T: gfx::Std430>(&mut self, data: &T) {
        std::ptr::copy_nonoverlapping(
            (data as *const T).cast(),
            self.ptr.add(self.offset),
            std::mem::size_of::<T>(),
        );
        self.offset += std::mem::size_of::<T>();
    }
}
