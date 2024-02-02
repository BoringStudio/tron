use std::ops::Range;

use bumpalo::Bump;
use glam::{IVec3, UVec3};
use shared::util::DeallocOnDrop;
use shared::FastHashSet;
use vulkanalia::prelude::v1_0::*;

use crate::device::{Device, WeakDevice};
use crate::resources::{
    Buffer, ClearValue, ComputePipeline, DescriptorSet, Filter, Framebuffer, GraphicsPipeline,
    Image, ImageLayout, ImageSubresourceLayers, ImageSubresourceRange, IndexType, LoadOp,
    PipelineBindPoint, PipelineLayout, PipelineStageFlags, Rect, ShaderStageFlags, Viewport,
};
use crate::types::OutOfDeviceMemory;
use crate::util::{compute_supported_access, FromGfx, ToVk};

/// Command buffer level.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CommandBufferLevel {
    Primary,
    Secondary,
}

impl FromGfx<CommandBufferLevel> for vk::CommandBufferLevel {
    fn from_gfx(value: CommandBufferLevel) -> Self {
        match value {
            CommandBufferLevel::Primary => Self::PRIMARY,
            CommandBufferLevel::Secondary => Self::SECONDARY,
        }
    }
}

/// A recorded sequence of commands that can be submitted to a queue.
pub struct CommandBuffer {
    inner: Box<Inner>,
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.as_ref();
        f.debug_struct("CommandBuffer")
            .field("handle", &inner.handle)
            .field("queue_family", &inner.queue_family)
            .field("level", &inner.level)
            .field("state", &inner.state)
            .finish()
    }
}

impl CommandBuffer {
    pub(crate) fn new(
        handle: vk::CommandBuffer,
        queue_family: u32,
        level: CommandBufferLevel,
        owner: Device,
    ) -> Self {
        Self {
            inner: Box::new(Inner {
                handle,
                queue_family,
                level,
                references: Default::default(),
                secondary_buffers: Default::default(),
                state: CommandBufferState::Full { owner },
                alloc: Bump::new(),
            }),
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.inner.handle
    }

    pub fn queue_family(&self) -> u32 {
        self.inner.queue_family
    }

    pub fn level(&self) -> CommandBufferLevel {
        self.inner.level
    }

    pub(crate) fn references(&self) -> &References {
        &self.inner.references
    }

    pub(crate) fn secondary_buffers(&self) -> &[CommandBuffer] {
        &self.inner.secondary_buffers
    }

    pub(crate) fn clear_references(&mut self) {
        self.inner.references.clear();
    }

    pub(crate) fn drain_secondary_buffers(&mut self) -> std::vec::Drain<'_, CommandBuffer> {
        self.inner.secondary_buffers.drain(..)
    }

    pub fn begin(&mut self) -> Result<(), OutOfDeviceMemory> {
        let inner = self.inner.as_mut();

        let device;
        let device = match &inner.state {
            CommandBufferState::Full { owner } => owner,
            CommandBufferState::Finished { owner } => {
                let Some(owner) = owner.upgrade() else {
                    return Ok(());
                };
                inner.state = CommandBufferState::Full {
                    owner: owner.clone(),
                };
                device = owner;
                &device
            }
        };

        let mut info = vk::CommandBufferBeginInfo::builder();

        let inheritance;
        match inner.level {
            CommandBufferLevel::Primary => {
                info = info.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            }
            CommandBufferLevel::Secondary => {
                inheritance = vk::CommandBufferInheritanceInfo::builder();
                info = info.inheritance_info(&inheritance)
            }
        }

        unsafe { device.logical().begin_command_buffer(inner.handle, &info) }
            .map_err(OutOfDeviceMemory::on_creation)
    }

    pub fn end(&mut self) -> Result<(), OutOfDeviceMemory> {
        let inner = self.inner.as_mut();

        let device = match &inner.state {
            CommandBufferState::Full { owner } => owner,
            CommandBufferState::Finished { .. } => return Ok(()),
        };

        unsafe { device.logical().end_command_buffer(inner.handle) }
            .map_err(OutOfDeviceMemory::on_creation)?;
        inner.state = CommandBufferState::Finished {
            owner: device.downgrade(),
        };
        Ok(())
    }

    pub(crate) fn execute_commands<I>(&mut self, buffers: I)
    where
        I: IntoIterator<Item = CommandBuffer>,
    {
        assert!(
            self.level() == CommandBufferLevel::Primary,
            "only primary command buffers can execute secondary command buffers"
        );

        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            let secondaty_buffers_offset = inner.secondary_buffers.len();
            inner.secondary_buffers.extend(buffers);
            if inner.secondary_buffers.len() == secondaty_buffers_offset {
                return;
            }

            let alloc = DeallocOnDrop(&mut inner.alloc);

            let handles = alloc.alloc_slice_fill_iter(
                inner.secondary_buffers[secondaty_buffers_offset..]
                    .iter()
                    .map(|b| {
                        assert!(
                            b.level() == CommandBufferLevel::Secondary,
                            "only secondary command buffers can be executed as commands"
                        );

                        b.handle()
                    }),
            );

            unsafe { device.logical().cmd_execute_commands(inner.handle, handles) }
        }
    }

    pub(crate) fn begin_render_pass(&mut self, framebuffer: &Framebuffer, clear: &[ClearValue]) {
        let inner = self.inner.as_mut();
        let Some(device) = inner.state.device_from_full() else {
            return;
        };
        let logical = device.logical();

        let alloc = DeallocOnDrop(&mut inner.alloc);

        inner.references.framebuffers.push(framebuffer.clone());

        let pass = &framebuffer.info().render_pass;

        let mut clear = clear.iter();
        let clear_values =
            alloc.alloc_slice_fill_iter(pass.info().attachments.iter().map(|attachment| {
                if attachment.load_op == LoadOp::Clear(()) {
                    clear
                        .next()
                        .expect("not enough clear values")
                        .try_to_vk(attachment.format)
                        .expect("invalid clear value")
                } else {
                    vk::ClearValue::default()
                }
            }));

        assert!(clear.next().is_none(), "too many clear values");

        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(pass.handle())
            .framebuffer(framebuffer.handle())
            .clear_values(clear_values)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: framebuffer.info().extent.to_vk(),
            });

        unsafe { logical.cmd_begin_render_pass(inner.handle, &info, vk::SubpassContents::INLINE) };
    }

    pub(crate) fn end_render_pass(&mut self) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            unsafe { device.logical().cmd_end_render_pass(inner.handle) }
        }
    }

    pub(crate) fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.graphics_pipelines.push(pipeline.clone());

            unsafe {
                device.logical().cmd_bind_pipeline(
                    inner.handle,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.handle(),
                )
            }
        }
    }

    pub(crate) fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.compute_pipelines.push(pipeline.clone());

            unsafe {
                device.logical().cmd_bind_pipeline(
                    inner.handle,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.handle(),
                )
            }
        }
    }

    pub(crate) fn bind_descriptor_sets(
        &mut self,
        bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.pipeline_layouts.insert(layout.clone());
            for &set in descriptor_sets {
                inner.references.descriptor_sets.push(set.clone());
            }

            let alloc = DeallocOnDrop(&mut inner.alloc);
            let descriptor_sets =
                alloc.alloc_slice_fill_iter(descriptor_sets.iter().map(|set| set.handle()));

            unsafe {
                device.logical().cmd_bind_descriptor_sets(
                    inner.handle,
                    bind_point.to_vk(),
                    layout.handle(),
                    first_set,
                    descriptor_sets,
                    dynamic_offsets,
                )
            }
        }
    }

    pub(crate) fn set_viewport(&mut self, viewport: &Viewport) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            let logical = device.logical();
            let viewport = vk::Viewport::from_gfx(*viewport);
            unsafe { logical.cmd_set_viewport(inner.handle, 0, std::slice::from_ref(&viewport)) }
        }
    }

    pub(crate) fn set_scissor(&mut self, scissor: &Rect) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            let logical = device.logical();
            let scissor = vk::Rect2D::from_gfx(*scissor);
            unsafe { logical.cmd_set_scissor(inner.handle, 0, std::slice::from_ref(&scissor)) }
        }
    }

    pub(crate) fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            unsafe {
                device.logical().cmd_draw(
                    inner.handle,
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
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            unsafe {
                device.logical().cmd_draw_indexed(
                    inner.handle,
                    indices.end - indices.start,
                    instances.end - instances.start,
                    indices.start,
                    vertex_offset,
                    instances.start,
                )
            }
        }
    }

    pub(crate) fn update_buffer(&mut self, buffer: &Buffer, offset: usize, data: &[u8]) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            assert!(offset % 4 == 0, "unaligned buffer offset");
            assert!(data.len() % 4 == 0, "unaligned buffer data length");
            assert!(data.len() <= 65536, "too much data to update");

            inner.references.buffers.insert(buffer.clone());

            let logical = device.logical();
            unsafe {
                logical.cmd_update_buffer(inner.handle, buffer.handle(), offset as u64, data)
            };
        }
    }

    pub(crate) fn bind_vertex_buffers(&mut self, first_binding: u32, buffers: &[(&Buffer, usize)]) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            for &(buffer, _) in buffers {
                inner.references.buffers.insert(buffer.clone());
            }

            let alloc = DeallocOnDrop(&mut inner.alloc);

            let offsets =
                alloc.alloc_slice_fill_iter(buffers.iter().map(|&(_, offset)| offset as u64));
            let buffers =
                alloc.alloc_slice_fill_iter(buffers.iter().map(|&(buffer, _)| buffer.handle()));

            let logical = device.logical();
            unsafe {
                logical.cmd_bind_vertex_buffers(inner.handle, first_binding, buffers, offsets)
            };
        }
    }

    pub(crate) fn bind_index_buffer(
        &mut self,
        buffer: &Buffer,
        offset: usize,
        index_type: IndexType,
    ) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.buffers.insert(buffer.clone());

            unsafe {
                device.logical().cmd_bind_index_buffer(
                    inner.handle,
                    buffer.handle(),
                    offset as u64,
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
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.buffers.insert(src_buffer.clone());
            inner.references.buffers.insert(dst_buffer.clone());

            let alloc = DeallocOnDrop(&mut inner.alloc);

            let regions =
                alloc.alloc_slice_fill_iter(regions.iter().map(|r| vk::BufferCopy::from_gfx(*r)));

            unsafe {
                device.logical().cmd_copy_buffer(
                    inner.handle,
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
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.images.push(src_image.clone());
            inner.references.images.push(dst_image.clone());

            let alloc = DeallocOnDrop(&mut inner.alloc);

            let regions =
                alloc.alloc_slice_fill_iter(regions.iter().map(|r| vk::ImageCopy::from_gfx(*r)));

            unsafe {
                device.logical().cmd_copy_image(
                    inner.handle,
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
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.buffers.insert(src_buffer.clone());
            inner.references.images.push(dst_image.clone());

            let alloc = DeallocOnDrop(&mut inner.alloc);

            let regions = alloc
                .alloc_slice_fill_iter(regions.iter().map(|r| vk::BufferImageCopy::from_gfx(*r)));

            unsafe {
                device.logical().cmd_copy_buffer_to_image(
                    inner.handle,
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
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.images.push(src_image.clone());
            inner.references.images.push(dst_image.clone());

            let alloc = DeallocOnDrop(&mut inner.alloc);

            let regions =
                alloc.alloc_slice_fill_iter(regions.iter().map(|r| vk::ImageBlit::from_gfx(*r)));

            unsafe {
                device.logical().cmd_blit_image(
                    inner.handle,
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
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            for item in image_memory_barriers {
                inner.references.images.push(item.image.clone());
            }
            for item in buffer_memory_barriers {
                inner.references.buffers.insert(item.buffer.clone());
            }

            let alloc = DeallocOnDrop(&mut inner.alloc);

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
                        .offset(b.offset as u64)
                        .size(b.size as u64)
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
                    inner.handle,
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
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            inner.references.pipeline_layouts.insert(layout.clone());

            unsafe {
                device.logical().cmd_push_constants(
                    inner.handle,
                    layout.handle(),
                    stages.to_vk(),
                    offset,
                    data,
                )
            }
        }
    }

    pub(crate) fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let inner = self.inner.as_mut();
        if let Some(device) = inner.state.device_from_full() {
            unsafe { device.logical().cmd_dispatch(inner.handle, x, y, z) }
        }
    }
}

struct Inner {
    handle: vk::CommandBuffer,
    queue_family: u32,
    level: CommandBufferLevel,
    references: References,
    secondary_buffers: Vec<CommandBuffer>,
    state: CommandBufferState,
    alloc: Bump,
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
    buffers: FastHashSet<Buffer>,
    images: Vec<Image>,
    framebuffers: Vec<Framebuffer>,
    graphics_pipelines: Vec<GraphicsPipeline>,
    compute_pipelines: Vec<ComputePipeline>,
    pipeline_layouts: FastHashSet<PipelineLayout>,
    descriptor_sets: Vec<DescriptorSet>,
}

impl References {
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
            && self.images.is_empty()
            && self.framebuffers.is_empty()
            && self.graphics_pipelines.is_empty()
            && self.compute_pipelines.is_empty()
            && self.pipeline_layouts.is_empty()
            && self.descriptor_sets.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
        self.images.clear();
        self.framebuffers.clear();
        self.graphics_pipelines.clear();
        self.compute_pipelines.clear();
        self.pipeline_layouts.clear();
        self.descriptor_sets.clear();
    }
}

/// Structure specifying a buffer copy operation.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferCopy {
    pub src_offset: usize,
    pub dst_offset: usize,
    pub size: usize,
}

impl FromGfx<BufferCopy> for vk::BufferCopy {
    #[inline]
    fn from_gfx(value: BufferCopy) -> Self {
        Self {
            src_offset: value.src_offset as u64,
            dst_offset: value.dst_offset as u64,
            size: value.size as u64,
        }
    }
}

/// Structure specifying an image copy operation.
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

/// Structure specifying a buffer image copy operation.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferImageCopy {
    pub buffer_offset: usize,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: ImageSubresourceLayers,
    pub image_offset: IVec3,
    pub image_extent: UVec3,
}

impl FromGfx<BufferImageCopy> for vk::BufferImageCopy {
    fn from_gfx(value: BufferImageCopy) -> Self {
        Self {
            buffer_offset: value.buffer_offset as u64,
            buffer_row_length: value.buffer_row_length,
            buffer_image_height: value.buffer_image_height,
            image_subresource: value.image_subresource.to_vk(),
            image_offset: value.image_offset.to_vk(),
            image_extent: value.image_extent.to_vk(),
        }
    }
}

/// Structure specifying an image blit operation.
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

/// Structure specifying a global memory barrier.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MemoryBarrier {
    pub src: AccessFlags,
    pub dst: AccessFlags,
}

/// Structure specifying a buffer memory barrier.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BufferMemoryBarrier<'a> {
    pub buffer: &'a Buffer,
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
    pub family_transfer: Option<(u32, u32)>,
    pub offset: usize,
    pub size: usize,
}

/// Structure specifying the parameters of an image memory barrier.
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

/// Structure specifying an image layout transition operation.
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
    /// Bitmask specifying memory access types that will participate in a memory dependency.
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
