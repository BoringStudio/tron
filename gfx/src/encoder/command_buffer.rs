use std::ops::Range;

use anyhow::Result;
use bumpalo::Bump;
use glam::{IVec3, UVec3};
use shared::FastHashSet;
use vulkanalia::prelude::v1_0::*;

use crate::device::{Device, WeakDevice};
use crate::queue::QueueId;
use crate::resources::{
    Buffer, ClearValue, ComputePipeline, Filter, Framebuffer, GraphicsPipeline, Image, ImageLayout,
    ImageSubresourceLayers, ImageSubresourceRange, IndexType, LoadOp, PipelineLayout,
    PipelineStageFlags, Rect, ShaderStageFlags, Viewport,
};
use crate::util::{compute_supported_access, DeallocOnDrop, FromGfx, ToVk};

/// A recorded sequence of commands that can be submitted to a queue.
pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    queue_id: QueueId,
    references: References,
    state: CommandBufferState,
    alloc: Bump,
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("handle", &self.handle)
            .field("queue_id", &self.queue_id)
            .field("state", &self.state)
            .finish()
    }
}

impl CommandBuffer {
    pub(crate) fn new(handle: vk::CommandBuffer, queue_id: QueueId, owner: Device) -> Self {
        Self {
            handle,
            queue_id,
            references: Default::default(),
            state: CommandBufferState::Full { owner },
            alloc: Bump::new(),
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn queue_id(&self) -> QueueId {
        self.queue_id
    }

    pub(crate) fn references(&self) -> &References {
        &self.references
    }

    pub(crate) fn references_mut(&mut self) -> &mut References {
        &mut self.references
    }

    pub fn begin(&mut self) -> Result<()> {
        let device;
        let device = match &self.state {
            CommandBufferState::Full { owner } => owner,
            CommandBufferState::Finished { owner } => {
                let Some(owner) = owner.upgrade() else {
                    return Ok(());
                };
                self.state = CommandBufferState::Full {
                    owner: owner.clone(),
                };
                device = owner;
                &device
            }
        };

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.logical().begin_command_buffer(self.handle, &info) }?;
        Ok(())
    }

    pub fn end(&mut self) -> Result<()> {
        let device = match &self.state {
            CommandBufferState::Full { owner } => owner,
            CommandBufferState::Finished { .. } => return Ok(()),
        };

        unsafe { device.logical().end_command_buffer(self.handle) }?;
        self.state = CommandBufferState::Finished {
            owner: device.downgrade(),
        };
        Ok(())
    }

    pub(crate) fn begin_render_pass(
        &mut self,
        framebuffer: &Framebuffer,
        clear: &[ClearValue],
    ) -> Result<()> {
        let Some(device) = self.state.device_from_full() else {
            return Ok(());
        };
        let logical = device.logical();

        let alloc = DeallocOnDrop(&mut self.alloc);

        self.references.framebuffers.push(framebuffer.clone());

        let pass = &framebuffer.info().render_pass;

        let mut clear = clear.iter();
        let mut clear_values_invalid = false;
        let clear_values =
            alloc.alloc_slice_fill_iter(pass.info().attachments.iter().map(|attachment| {
                if attachment.load_op == LoadOp::Clear(()) {
                    if let Some(clear) = clear.next().and_then(|v| v.try_to_vk(attachment.format)) {
                        return clear;
                    } else {
                        clear_values_invalid = true;
                    }
                }

                vk::ClearValue::default()
            }));
        anyhow::ensure!(
            !clear_values_invalid && clear.next().is_none(),
            "clear values are invalid"
        );

        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(pass.handle())
            .framebuffer(framebuffer.handle())
            .clear_values(clear_values)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: framebuffer.info().extent.to_vk(),
            });

        unsafe { logical.cmd_begin_render_pass(self.handle, &info, vk::SubpassContents::INLINE) }
        Ok(())
    }

    pub(crate) fn end_render_pass(&mut self) {
        if let Some(device) = self.state.device_from_full() {
            unsafe { device.logical().cmd_end_render_pass(self.handle) }
        }
    }

    pub(crate) fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        if let Some(device) = self.state.device_from_full() {
            self.references.graphics_pipelines.push(pipeline.clone());

            unsafe {
                device.logical().cmd_bind_pipeline(
                    self.handle,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.handle(),
                )
            }
        }
    }

    pub(crate) fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        if let Some(device) = self.state.device_from_full() {
            self.references.compute_pipelines.push(pipeline.clone());

            unsafe {
                device.logical().cmd_bind_pipeline(
                    self.handle,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.handle(),
                )
            }
        }
    }

    pub(crate) fn set_viewport(&mut self, viewport: &Viewport) {
        if let Some(device) = self.state.device_from_full() {
            let logical = device.logical();
            let viewport = vk::Viewport::from_gfx(*viewport);
            unsafe { logical.cmd_set_viewport(self.handle, 0, std::slice::from_ref(&viewport)) }
        }
    }

    pub(crate) fn set_scissor(&mut self, scissor: &Rect) {
        if let Some(device) = self.state.device_from_full() {
            let logical = device.logical();
            let scissor = vk::Rect2D::from_gfx(*scissor);
            unsafe { logical.cmd_set_scissor(self.handle, 0, std::slice::from_ref(&scissor)) }
        }
    }

    pub(crate) fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        if let Some(device) = self.state.device_from_full() {
            unsafe {
                device.logical().cmd_draw(
                    self.handle,
                    vertices.end - vertices.start,
                    instances.end - instances.start,
                    vertices.start,
                    instances.start,
                )
            }
        }
    }

    pub(crate) fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        vertex_offset: i32,
        instances: Range<u32>,
    ) {
        if let Some(device) = self.state.device_from_full() {
            unsafe {
                device.logical().cmd_draw_indexed(
                    self.handle,
                    indices.end - indices.start,
                    instances.end - instances.start,
                    indices.start,
                    vertex_offset,
                    instances.start,
                )
            }
        }
    }

    pub(crate) fn update_buffer(
        &mut self,
        buffer: &Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<()> {
        if let Some(device) = self.state.device_from_full() {
            anyhow::ensure!(offset % 4 == 0, "unaligned buffer offset");
            anyhow::ensure!(data.len() % 4 == 0, "unaligned buffer data length");
            anyhow::ensure!(data.len() <= 65536, "too much data to update");

            self.references.buffers.push(buffer.clone());

            let logical = device.logical();
            unsafe { logical.cmd_update_buffer(self.handle, buffer.handle(), offset, data) }
        }
        Ok(())
    }

    pub(crate) fn bind_vertex_buffers(&mut self, first: u32, buffers: &[(&Buffer, u64)]) {
        if let Some(device) = self.state.device_from_full() {
            for &(buffer, _) in buffers {
                self.references.buffers.push(buffer.clone());
            }

            let alloc = DeallocOnDrop(&mut self.alloc);

            let offsets = alloc.alloc_slice_fill_iter(buffers.iter().map(|&(_, offset)| offset));
            let buffers =
                alloc.alloc_slice_fill_iter(buffers.iter().map(|&(buffer, _)| buffer.handle()));

            let logical = device.logical();
            unsafe { logical.cmd_bind_vertex_buffers(self.handle, first, buffers, offsets) };
        }
    }

    pub(crate) fn bind_index_buffer(
        &mut self,
        buffer: &Buffer,
        offset: u64,
        index_type: IndexType,
    ) {
        if let Some(device) = self.state.device_from_full() {
            self.references.buffers.push(buffer.clone());

            unsafe {
                device.logical().cmd_bind_index_buffer(
                    self.handle,
                    buffer.handle(),
                    offset,
                    index_type.to_vk(),
                )
            }
        }
    }

    pub(crate) fn copy_buffer(
        &mut self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        regions: &[BufferCopy],
    ) {
        if let Some(device) = self.state.device_from_full() {
            self.references.buffers.push(src_buffer.clone());
            self.references.buffers.push(dst_buffer.clone());

            let alloc = DeallocOnDrop(&mut self.alloc);

            let regions = alloc.alloc_slice_fill_iter(regions.iter().map(|r| {
                vk::BufferCopy::builder()
                    .src_offset(r.src_offset)
                    .dst_offset(r.dst_offset)
                    .size(r.size)
            }));

            unsafe {
                device.logical().cmd_copy_buffer(
                    self.handle,
                    src_buffer.handle(),
                    dst_buffer.handle(),
                    regions,
                )
            }
        }
    }

    pub(crate) fn copy_image(
        &mut self,
        src_image: &Image,
        src_layout: ImageLayout,
        dst_image: &Image,
        dst_layout: ImageLayout,
        regions: &[ImageCopy],
    ) {
        if let Some(device) = self.state.device_from_full() {
            self.references.images.push(src_image.clone());
            self.references.images.push(dst_image.clone());

            let alloc = DeallocOnDrop(&mut self.alloc);

            let regions =
                alloc.alloc_slice_fill_iter(regions.iter().map(|r| vk::ImageCopy::from_gfx(*r)));

            unsafe {
                device.logical().cmd_copy_image(
                    self.handle,
                    src_image.handle(),
                    src_layout.to_vk(),
                    dst_image.handle(),
                    dst_layout.to_vk(),
                    regions,
                )
            }
        }
    }

    pub(crate) fn copy_buffer_to_image(
        &mut self,
        src_buffer: &Buffer,
        dst_image: &Image,
        dst_layout: ImageLayout,
        regions: &[BufferImageCopy],
    ) {
        if let Some(device) = self.state.device_from_full() {
            self.references.buffers.push(src_buffer.clone());
            self.references.images.push(dst_image.clone());

            let alloc = DeallocOnDrop(&mut self.alloc);

            let regions = alloc
                .alloc_slice_fill_iter(regions.iter().map(|r| vk::BufferImageCopy::from_gfx(*r)));

            unsafe {
                device.logical().cmd_copy_buffer_to_image(
                    self.handle,
                    src_buffer.handle(),
                    dst_image.handle(),
                    dst_layout.to_vk(),
                    regions,
                )
            }
        }
    }

    pub(crate) fn blit_image(
        &mut self,
        src_image: &Image,
        src_layout: ImageLayout,
        dst_image: &Image,
        dst_layout: ImageLayout,
        regions: &[ImageBlit],
        filter: Filter,
    ) {
        if let Some(device) = self.state.device_from_full() {
            self.references.images.push(src_image.clone());
            self.references.images.push(dst_image.clone());

            let alloc = DeallocOnDrop(&mut self.alloc);

            let regions =
                alloc.alloc_slice_fill_iter(regions.iter().map(|r| vk::ImageBlit::from_gfx(*r)));

            unsafe {
                device.logical().cmd_blit_image(
                    self.handle,
                    src_image.handle(),
                    src_layout.to_vk(),
                    dst_image.handle(),
                    dst_layout.to_vk(),
                    regions,
                    filter.to_vk(),
                )
            }
        }
    }

    pub(crate) fn pipeline_barrier(
        &mut self,
        src: PipelineStageFlags,
        dst: PipelineStageFlags,
        memory_barrier: Option<MemoryBarrier>,
        buffer_memory_barriers: &[BufferMemoryBarrier],
        image_memory_barriers: &[ImageMemoryBarrier],
    ) {
        if let Some(device) = self.state.device_from_full() {
            for item in image_memory_barriers {
                self.references.images.push(item.image.clone());
            }
            for item in buffer_memory_barriers {
                self.references.buffers.push(item.buffer.clone());
            }

            let alloc = DeallocOnDrop(&mut self.alloc);

            let memory_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(
                    memory_barrier.map_or(compute_supported_access(src.to_vk()), |b| b.src.to_vk()),
                )
                .dst_access_mask(
                    memory_barrier.map_or(compute_supported_access(dst.to_vk()), |b| b.dst.to_vk()),
                );

            let buffer_memory_barriers =
                alloc.alloc_slice_fill_iter(buffer_memory_barriers.iter().map(|b| {
                    vk::BufferMemoryBarrier::builder()
                        .buffer(b.buffer.handle())
                        .offset(b.offset)
                        .size(b.size)
                        .src_access_mask(b.src_access.to_vk())
                        .dst_access_mask(b.dst_access.to_vk())
                        .src_queue_family_index(
                            b.family_transfer
                                .map(|v| v.0)
                                .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                        )
                        .dst_queue_family_index(
                            b.family_transfer
                                .map(|v| v.1)
                                .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                        )
                }));

            let image_memory_barriers =
                alloc.alloc_slice_fill_iter(image_memory_barriers.iter().map(|b| {
                    vk::ImageMemoryBarrier::builder()
                        .image(b.image.handle())
                        .src_access_mask(b.src_access.to_vk())
                        .dst_access_mask(b.dst_access.to_vk())
                        .old_layout(b.old_layout.to_vk())
                        .new_layout(b.new_layout.to_vk())
                        .src_queue_family_index(
                            b.family_transfer
                                .map(|v| v.0)
                                .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                        )
                        .dst_queue_family_index(
                            b.family_transfer
                                .map(|v| v.1)
                                .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                        )
                        .subresource_range(vk::ImageSubresourceRange::from_gfx(b.subresource_range))
                }));

            unsafe {
                device.logical().cmd_pipeline_barrier(
                    self.handle,
                    src.to_vk(),
                    dst.to_vk(),
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&memory_barrier),
                    buffer_memory_barriers,
                    image_memory_barriers,
                )
            }
        }
    }

    pub(crate) fn push_constants(
        &mut self,
        layout: &PipelineLayout,
        stages: ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        if let Some(device) = self.state.device_from_full() {
            self.references.pipeline_layouts.insert(layout.clone());

            unsafe {
                device.logical().cmd_push_constants(
                    self.handle,
                    layout.handle(),
                    stages.to_vk(),
                    offset,
                    data,
                )
            }
        }
    }

    pub(crate) fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        if let Some(device) = self.state.device_from_full() {
            unsafe { device.logical().cmd_dispatch(self.handle, x, y, z) }
        }
    }
}

#[derive(Debug)]
enum CommandBufferState {
    Full { owner: Device },
    Finished { owner: WeakDevice },
}

impl CommandBufferState {
    fn device_from_full(&self) -> Option<&Device> {
        match self {
            Self::Full { owner } => Some(owner),
            Self::Finished { .. } => None,
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct References {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    framebuffers: Vec<Framebuffer>,
    graphics_pipelines: Vec<GraphicsPipeline>,
    compute_pipelines: Vec<ComputePipeline>,
    pipeline_layouts: FastHashSet<PipelineLayout>,
}

impl References {
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
            && self.images.is_empty()
            && self.framebuffers.is_empty()
            && self.graphics_pipelines.is_empty()
            && self.compute_pipelines.is_empty()
            && self.pipeline_layouts.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
        self.images.clear();
        self.framebuffers.clear();
        self.graphics_pipelines.clear();
        self.compute_pipelines.clear();
        self.pipeline_layouts.clear();
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferCopy {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

impl FromGfx<BufferCopy> for vk::BufferCopy {
    #[inline]
    fn from_gfx(value: BufferCopy) -> Self {
        Self {
            src_offset: value.src_offset,
            dst_offset: value.dst_offset,
            size: value.size,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageCopy {
    pub src_subresource: ImageSubresourceLayers,
    pub src_offset: IVec3,
    pub dst_subresource: ImageSubresourceLayers,
    pub dst_offset: IVec3,
    pub extent: UVec3,
}

impl FromGfx<ImageCopy> for vk::ImageCopy {
    fn from_gfx(value: ImageCopy) -> Self {
        Self {
            src_subresource: value.src_subresource.to_vk(),
            src_offset: value.src_offset.to_vk(),
            dst_subresource: value.dst_subresource.to_vk(),
            dst_offset: value.dst_offset.to_vk(),
            extent: value.extent.to_vk(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferImageCopy {
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: ImageSubresourceLayers,
    pub image_offset: IVec3,
    pub image_extent: UVec3,
}

impl FromGfx<BufferImageCopy> for vk::BufferImageCopy {
    fn from_gfx(value: BufferImageCopy) -> Self {
        Self {
            buffer_offset: value.buffer_offset,
            buffer_row_length: value.buffer_row_length,
            buffer_image_height: value.buffer_image_height,
            image_subresource: value.image_subresource.to_vk(),
            image_offset: value.image_offset.to_vk(),
            image_extent: value.image_extent.to_vk(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageBlit {
    pub src_subresource: ImageSubresourceLayers,
    pub src_offsets: [IVec3; 2],
    pub dst_subresource: ImageSubresourceLayers,
    pub dst_offsets: [IVec3; 2],
}

impl FromGfx<ImageBlit> for vk::ImageBlit {
    fn from_gfx(value: ImageBlit) -> Self {
        Self {
            src_subresource: value.src_subresource.to_vk(),
            src_offsets: value.src_offsets.map(|v| v.to_vk()),
            dst_subresource: value.dst_subresource.to_vk(),
            dst_offsets: value.dst_offsets.map(|v| v.to_vk()),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MemoryBarrier {
    pub src: AccessFlags,
    pub dst: AccessFlags,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BufferMemoryBarrier<'a> {
    pub buffer: &'a Buffer,
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
    pub family_transfer: Option<(u32, u32)>,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageMemoryBarrier<'a> {
    pub image: &'a Image,
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
    pub old_layout: Option<ImageLayout>,
    pub new_layout: ImageLayout,
    pub family_transfer: Option<(u32, u32)>,
    pub subresource_range: ImageSubresourceRange,
}

impl<'a> ImageMemoryBarrier<'a> {
    pub fn transition_whole(
        image: &'a Image,
        access: Range<AccessFlags>,
        layout: Range<ImageLayout>,
    ) -> Self {
        Self {
            image,
            src_access: access.start,
            dst_access: access.end,
            old_layout: Some(layout.start),
            new_layout: layout.end,
            family_transfer: None,
            subresource_range: ImageSubresourceRange::whole(image.info()),
        }
    }

    pub fn initialize_whole(image: &'a Image, access: AccessFlags, layout: ImageLayout) -> Self {
        Self {
            image,
            src_access: AccessFlags::empty(),
            dst_access: access,
            old_layout: None,
            new_layout: layout,
            family_transfer: None,
            subresource_range: ImageSubresourceRange::whole(image.info()),
        }
    }
}

impl<'a> From<ImageLayoutTransition<'a>> for ImageMemoryBarrier<'a> {
    fn from(value: ImageLayoutTransition<'a>) -> Self {
        Self {
            image: value.image,
            src_access: value.src_access,
            dst_access: value.dst_access,
            old_layout: value.old_layout,
            new_layout: value.new_layout,
            family_transfer: None,
            subresource_range: value.subresource_range,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageLayoutTransition<'a> {
    pub image: &'a Image,
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
    pub old_layout: Option<ImageLayout>,
    pub new_layout: ImageLayout,
    pub subresource_range: ImageSubresourceRange,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct AccessFlags: u32 {
        const INDIRECT_COMMAND_READ = 1;
        const INDEX_READ = 1 << 1;
        const VERTEX_ATTRIBUTE_READ = 1 << 2;
        const UNIFORM_READ = 1 << 3;
        const INPUT_ATTACHMENT_READ = 1 << 4;
        const SHADER_READ = 1 << 5;
        const SHADER_WRITE = 1 << 6;
        const COLOR_ATTACHMENT_READ = 1 << 7;
        const COLOR_ATTACHMENT_WRITE = 1 << 8;
        const DEPTH_STENCIL_ATTACHMENT_READ = 1 << 9;
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 1 << 10;
        const TRANSFER_READ = 1 << 11;
        const TRANSFER_WRITE = 1 << 12;
        const HOST_READ = 1 << 13;
        const HOST_WRITE = 1 << 14;
        const MEMORY_READ = 1 << 15;
        const MEMORY_WRITE = 1 << 16;
    }
}

impl FromGfx<AccessFlags> for vk::AccessFlags {
    fn from_gfx(value: AccessFlags) -> Self {
        let mut res = Self::empty();
        if value.contains(AccessFlags::INDIRECT_COMMAND_READ) {
            res |= Self::INDIRECT_COMMAND_READ;
        }
        if value.contains(AccessFlags::INDEX_READ) {
            res |= Self::INDEX_READ;
        }
        if value.contains(AccessFlags::VERTEX_ATTRIBUTE_READ) {
            res |= Self::VERTEX_ATTRIBUTE_READ;
        }
        if value.contains(AccessFlags::UNIFORM_READ) {
            res |= Self::UNIFORM_READ;
        }
        if value.contains(AccessFlags::INPUT_ATTACHMENT_READ) {
            res |= Self::INPUT_ATTACHMENT_READ;
        }
        if value.contains(AccessFlags::SHADER_READ) {
            res |= Self::SHADER_READ;
        }
        if value.contains(AccessFlags::SHADER_WRITE) {
            res |= Self::SHADER_WRITE;
        }
        if value.contains(AccessFlags::COLOR_ATTACHMENT_READ) {
            res |= Self::COLOR_ATTACHMENT_READ;
        }
        if value.contains(AccessFlags::COLOR_ATTACHMENT_WRITE) {
            res |= Self::COLOR_ATTACHMENT_WRITE;
        }
        if value.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ) {
            res |= Self::DEPTH_STENCIL_ATTACHMENT_READ;
        }
        if value.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
            res |= Self::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        if value.contains(AccessFlags::TRANSFER_READ) {
            res |= Self::TRANSFER_READ;
        }
        if value.contains(AccessFlags::TRANSFER_WRITE) {
            res |= Self::TRANSFER_WRITE;
        }
        if value.contains(AccessFlags::HOST_READ) {
            res |= Self::HOST_READ;
        }
        if value.contains(AccessFlags::HOST_WRITE) {
            res |= Self::HOST_WRITE;
        }
        if value.contains(AccessFlags::MEMORY_READ) {
            res |= Self::MEMORY_READ;
        }
        if value.contains(AccessFlags::MEMORY_WRITE) {
            res |= Self::MEMORY_WRITE;
        }
        res
    }
}
