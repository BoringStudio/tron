use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use range_alloc::RangeAllocator;

use crate::renderer::types::{Mesh, MeshHandle, RawMeshHandle, Vertex};
use crate::renderer::CommandEncoder;
use crate::util::ResourceRegistry;

pub struct MeshManager {
    buffers: MeshBuffers,
    vertex_alloc: RangeAllocator<usize>,
    index_alloc: RangeAllocator<usize>,
    registry: ResourceRegistry<InternalMesh, Mesh>,
}

impl MeshManager {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffers = MeshBuffers::new(device, STARTING_VERTEX_COUNT, STARTING_INDEX_COUNT);
        let vertex_alloc = RangeAllocator::new(0..STARTING_VERTEX_COUNT);
        let index_alloc = RangeAllocator::new(0..STARTING_INDEX_COUNT);
        let registry = ResourceRegistry::new();

        Self {
            buffers,
            vertex_alloc,
            index_alloc,
            registry,
        }
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

    pub fn allocate(counter: &AtomicUsize) -> MeshHandle {
        let idx = counter.fetch_add(1, Ordering::Relaxed);
        MeshHandle::new(idx)
    }

    pub fn get_mesh(&self, handle: RawMeshHandle) -> &InternalMesh {
        self.registry.get(handle)
    }

    pub fn set_mesh(&mut self, ctx: &mut CommandEncoder<'_>, handle: &MeshHandle, mesh: Mesh) {
        let vertex_count = mesh.vertices.len();
        let index_count = mesh.indices.len();

        let mut vertex_range = self.vertex_alloc.allocate_range(vertex_count).ok();
        let mut index_range = self.index_alloc.allocate_range(index_count).ok();

        let requested = match (&vertex_range, &index_range) {
            (None, None) => Some((vertex_count, index_count)),
            (None, Some(_)) => Some((vertex_count, 0)),
            (Some(_), None) => Some((0, index_count)),
            (Some(_), Some(_)) => None,
        };

        if let Some((remaining_vertices, remaining_indices)) = requested {
            self.realloc_buffers(
                ctx.device,
                &mut ctx.encoder,
                remaining_vertices,
                remaining_indices,
            );
            vertex_range = self.vertex_alloc.allocate_range(vertex_count).ok();
            index_range = self.index_alloc.allocate_range(index_count).ok();
        }

        let vertex_range = vertex_range.unwrap();
        let index_range = index_range.unwrap();

        ctx.queue.write_buffer(
            &self.buffers.vertices,
            (vertex_range.start * VERTEX_SIZE) as wgpu::BufferAddress,
            bytemuck::cast_slice(&mesh.vertices),
        );
        ctx.queue.write_buffer(
            &self.buffers.indices,
            (index_range.start * INDEX_SIZE) as wgpu::BufferAddress,
            bytemuck::cast_slice(&mesh.indices),
        );

        let data = InternalMesh {
            vertex_range,
            index_range,
        };
        self.registry.insert(handle, data);
    }

    fn realloc_buffers(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        remaining_vertices: usize,
        remaining_indices: usize,
    ) {
        let new_vertex_count = (self.vertex_count() + remaining_vertices).next_power_of_two();
        let new_index_count = (self.index_count() + remaining_indices).next_power_of_two();

        let new_buffers = MeshBuffers::new(device, new_vertex_count, new_index_count);

        let mut new_vertex_alloc = RangeAllocator::new(0..new_vertex_count);
        let mut new_index_alloc = RangeAllocator::new(0..new_index_count);

        for mesh in self.registry.values_mut() {
            if mesh.index_range.is_empty() {
                continue;
            }

            let new_vertex_range = new_vertex_alloc
                .allocate_range(mesh.vertex_range.len())
                .unwrap();
            let new_index_range = new_index_alloc
                .allocate_range(mesh.index_range.len())
                .unwrap();

            new_buffers.copy_vertices_from(
                encoder,
                &self.buffers,
                &mesh.vertex_range,
                &new_vertex_range,
            );

            new_buffers.copy_indices_from(
                encoder,
                &self.buffers,
                &mesh.index_range,
                &new_index_range,
            );

            mesh.vertex_range = new_vertex_range;
            mesh.index_range = new_index_range;
        }

        self.buffers = new_buffers;
        self.vertex_alloc = new_vertex_alloc;
        self.index_alloc = new_index_alloc;
    }
}

pub struct InternalMesh {
    pub vertex_range: Range<usize>,
    pub index_range: Range<usize>,
}

impl InternalMesh {
    pub fn indices(&self) -> Range<u32> {
        Range {
            start: self.index_range.start as u32,
            end: self.index_range.end as u32,
        }
    }
}

pub struct MeshBuffers {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
}

impl MeshBuffers {
    pub fn new(device: &wgpu::Device, vertex_count: usize, index_count: usize) -> Self {
        let vertices_len = vertex_count * VERTEX_SIZE;
        let indices_len = index_count * INDEX_SIZE;

        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertices_buffer"),
            size: vertices_len as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indices_buffer"),
            size: indices_len as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { vertices, indices }
    }

    pub fn bind<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_vertex_buffer(VERTICES_SLOT, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
    }

    fn copy_vertices_from(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        from: &Self,
        src_range: &Range<usize>,
        dst_range: &Range<usize>,
    ) {
        copy_buffer_part(
            encoder,
            &from.vertices,
            &self.vertices,
            src_range,
            dst_range,
            VERTEX_SIZE,
        );
    }

    fn copy_indices_from(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        from: &Self,
        src_range: &Range<usize>,
        dst_range: &Range<usize>,
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
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    src_range: &Range<usize>,
    dst_range: &Range<usize>,
    item_size: usize,
) {
    let src_start = src_range.start * item_size;
    let len = src_range.end * item_size - src_start;
    let dst_start = dst_range.start * item_size;
    encoder.copy_buffer_to_buffer(
        src,
        src_start as wgpu::BufferAddress,
        dst,
        dst_start as wgpu::BufferAddress,
        len as wgpu::BufferAddress,
    );
}

pub const STARTING_VERTEX_COUNT: usize = 1 << 16;
pub const STARTING_INDEX_COUNT: usize = 1 << 16;

pub const VERTEX_SIZE: usize = std::mem::size_of::<Vertex>();
pub const INDEX_SIZE: usize = std::mem::size_of::<u32>();

pub const VERTICES_SLOT: u32 = 0;
