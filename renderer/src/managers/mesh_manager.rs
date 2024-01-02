use std::ops::Range;

use anyhow::Result;
use range_alloc::RangeAllocator;

use crate::resource_registry::ResourceRegistry;
use crate::types::Mesh;

pub struct MeshManager {
    buffers: MeshBuffers,
    vertex_alloc: RangeAllocator<usize>,
    index_alloc: RangeAllocator<usize>,
    registry: ResourceRegistry<InternalMesh, Mesh>,
}

impl MeshManager {
    pub fn new(device: &gfx::Device, vertex_size: usize) -> Result<Self> {
        const INITIAL_VERTEX_COUNT: usize = 1 << 16;
        const INITIAL_INDEX_COUNT: usize = 1 << 16;

        let buffers = MeshBuffers::new(
            device,
            INITIAL_INDEX_COUNT,
            vertex_size,
            INITIAL_INDEX_COUNT,
        )?;
        let vertex_alloc = RangeAllocator::new(0..INITIAL_VERTEX_COUNT);
        let index_alloc = RangeAllocator::new(0..INITIAL_INDEX_COUNT);

        Ok(Self {
            buffers,
            vertex_alloc,
            index_alloc,
            registry: Default::default(),
        })
    }

    pub fn buffers(&self) -> &MeshBuffers {
        &self.buffers
    }

    pub fn vertex_count(&self) -> usize {
        self.vertex_alloc.initial_range().end
    }

    pub fn index_count(&self) -> usize {
        self.index_alloc.initial_range().end
    }

    fn realloc(
        &mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        remaining_vertices: usize,
        remaining_indices: usize,
    ) {
        let new_vertex_count = (self.vertex_count() + remaining_vertices).next_power_of_two();
        let new_index_count = (self.index_count() + remaining_indices).next_power_of_two();

        let new_buffers = MeshBuffers::new(
            device,
            new_vertex_count,
            self.buffers.vertex_size,
            new_index_count,
        );

        let mut new_vertex_alloc = RangeAllocator::new(0..new_vertex_count);
        let mut new_index_alloc = RangeAllocator::new(0..new_index_count);
    }
}

pub struct InternalMesh {
    pub vertices_range: Range<usize>,
    pub indices_range: Range<usize>,
}

impl InternalMesh {
    pub fn indices(&self) -> Range<u32> {
        Range {
            start: self.indices_range.start as u32,
            end: self.indices_range.end as u32,
        }
    }
}

pub struct MeshBuffers {
    pub vertices: gfx::Buffer,
    pub indices: gfx::Buffer,
    pub vertex_size: usize,
}

impl MeshBuffers {
    fn new(
        device: &gfx::Device,
        vertex_count: usize,
        vertex_size: usize,
        index_count: usize,
    ) -> Result<Self> {
        let vertices_len = vertex_count * vertex_size;
        let indices_len = index_count * INDEX_SIZE;

        let vertices = device.create_buffer(gfx::BufferInfo {
            align: VERTEX_ALIGN_MASK,
            size: vertices_len as u64,
            usage: gfx::BufferUsage::TRANSFER_DST | gfx::BufferUsage::VERTEX,
        })?;

        let indices = device.create_buffer(gfx::BufferInfo {
            align: INDEX_ALIGN_MASK,
            size: indices_len as u64,
            usage: gfx::BufferUsage::TRANSFER_DST | gfx::BufferUsage::INDEX,
        })?;

        Ok(Self {
            vertices,
            indices,
            vertex_size,
        })
    }

    pub fn bind(&self, pass: &mut gfx::EncoderCommon) {
        pass.bind_vertex_buffers(0, &[(&self.vertices, 0)]);
        pass.bind_index_buffer(&self.indices, 0, gfx::IndexType::U32);
    }

    fn copy_vertices_from(
        &self,
        encoder: &mut gfx::Encoder,
        from: &Self,
        src_range: Range<usize>,
        dst_range: Range<usize>,
    ) {
        copy_buffer_part(
            encoder,
            &from.vertices,
            &self.vertices,
            src_range,
            dst_range,
            from.vertex_size,
        );
    }

    fn copy_indices_from(
        &self,
        encoder: &mut gfx::Encoder,
        from: &Self,
        src_range: Range<usize>,
        dst_range: Range<usize>,
    ) {
        copy_buffer_part(
            encoder,
            &from.indices,
            &self.indices,
            src_range,
            dst_range,
            INDEX_SIZE,
        );
    }
}

fn copy_buffer_part(
    encoder: &mut gfx::Encoder,
    src: &gfx::Buffer,
    dst: &gfx::Buffer,
    src_range: Range<usize>,
    dst_range: Range<usize>,
    item_size: usize,
) {
    let src_offset = src_range.start * item_size;
    let size = src_range.end * item_size - src_offset;
    let dst_offset = dst_range.start * item_size;
    encoder.copy_buffer(
        src,
        dst,
        &[gfx::BufferCopy {
            src_offset: src_offset as u64,
            dst_offset: dst_offset as u64,
            size: size as u64,
        }],
    );
}

const VERTEX_ALIGN_MASK: u64 = 0b1111;
const INDEX_ALIGN_MASK: u64 = 0b11;
const INDEX_SIZE: usize = std::mem::size_of::<u32>();
