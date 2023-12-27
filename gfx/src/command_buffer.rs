use vulkanalia::prelude::v1_0::*;

use anyhow::Result;
use bumpalo::Bump;
use glam::{IVec3, UVec3};

use crate::device::{Device, WeakDevice};
use crate::queue::QueueId;
use crate::resources::{
    Buffer, ClearValue, ComputePipeline, Framebuffer, GraphicsPipeline, Image, ImageLayout,
    ImageSubresourceLayers, IndexType, LoadOp,
};
use crate::util::{FromGfx, ToVk};

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    queue_id: QueueId,
    references: References,
    state: CommandBufferState,
    alloc: Bump,
}

impl CommandBuffer {
    pub fn new(handle: vk::CommandBuffer, queue_id: QueueId, owner: Device) -> Self {
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

    pub fn references(&self) -> &References {
        &self.references
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

    pub fn begin_render_pass(
        &mut self,
        framebuffer: &Framebuffer,
        clear: &[ClearValue],
    ) -> Result<()> {
        let Some(device) = self.state.device_from_full() else {
            return Ok(());
        };
        let logical = device.logical();

        self.references.framebuffers.push(framebuffer.clone());

        let pass = &framebuffer.info().render_pass;

        let mut clear = clear.iter();
        let mut clear_values_invalid = false;
        let clear_values = self
            .alloc
            .alloc_slice_fill_iter(pass.info().attachments.iter().map(|attachment| {
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

    pub fn end_render_pass(&mut self) {
        if let Some(device) = self.state.device_from_full() {
            unsafe { device.logical().cmd_end_render_pass(self.handle) }
        }
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
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

    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
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

    pub fn set_viewport(&mut self, viewport: &vk::Viewport) {
        if let Some(device) = self.state.device_from_full() {
            let logical = device.logical();
            unsafe { logical.cmd_set_viewport(self.handle, 0, std::slice::from_ref(viewport)) }
        }
    }

    pub fn set_scissor(&mut self, scissor: &vk::Rect2D) {
        if let Some(device) = self.state.device_from_full() {
            let logical = device.logical();
            unsafe { logical.cmd_set_scissor(self.handle, 0, std::slice::from_ref(scissor)) }
        }
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
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

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        vertex_offset: i32,
        instances: std::ops::Range<u32>,
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

    pub fn update_buffer(&mut self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<()> {
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

    pub fn bind_vertex_buffers(&mut self, first: u32, buffers: &[(&Buffer, u64)]) {
        if let Some(device) = self.state.device_from_full() {
            for &(buffer, _) in buffers {
                self.references.buffers.push(buffer.clone());
            }

            let offsets = self
                .alloc
                .alloc_slice_fill_iter(buffers.iter().map(|&(_, offset)| offset));
            let buffers = self
                .alloc
                .alloc_slice_fill_iter(buffers.iter().map(|&(buffer, _)| buffer.handle()));

            let logical = device.logical();
            unsafe { logical.cmd_bind_vertex_buffers(self.handle, first, buffers, offsets) };

            self.alloc.reset();
        }
    }

    pub fn bind_index_buffer(&mut self, buffer: &Buffer, offset: u64, index_type: IndexType) {
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

    pub fn copy_buffer(
        &mut self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        regions: &[BufferCopy],
    ) {
        if let Some(device) = self.state.device_from_full() {
            self.references.buffers.push(src_buffer.clone());
            self.references.buffers.push(dst_buffer.clone());

            let regions = self.alloc.alloc_slice_fill_iter(regions.iter().map(|r| {
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

            self.alloc.reset();
        }
    }

    pub fn copy_image(
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

            let regions = self
                .alloc
                .alloc_slice_fill_iter(regions.iter().map(|r| vk::ImageCopy::from_gfx(*r)));

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

            self.alloc.reset();
        }
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
pub struct References {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    framebuffers: Vec<Framebuffer>,
    graphics_pipelines: Vec<GraphicsPipeline>,
    compute_pipelines: Vec<ComputePipeline>,
}

impl References {
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.framebuffers.clear();
    }
}
