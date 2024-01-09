use std::ops::Range;

use anyhow::Result;

pub use self::command_buffer::*;
use crate::device::Device;
use crate::queue::QueueFlags;
use crate::resources::{
    Buffer, BufferInfo, BufferUsage, ClearValue, ComputePipeline, DescriptorSet, Filter,
    Framebuffer, GraphicsPipeline, Image, ImageLayout, IndexType, PipelineBindPoint,
    PipelineLayout, Rect, RenderPass, ShaderStageFlags, Viewport,
};
use crate::PipelineStageFlags;

mod command_buffer;

/// A command buffer encoder.
pub struct Encoder {
    inner: EncoderCommon,
    guard: EncoderDropGuard,
}

impl Encoder {
    pub(crate) fn new(command_buffer: CommandBuffer, capabilities: QueueFlags) -> Self {
        Self {
            inner: EncoderCommon {
                command_buffer,
                capabilities,
            },
            guard: EncoderDropGuard,
        }
    }

    /// Finish recording the command buffer.
    pub fn finish(mut self) -> Result<CommandBuffer> {
        std::mem::forget(self.guard);
        self.inner.command_buffer.end()?;
        Ok(self.inner.command_buffer)
    }

    /// Discard the command buffer.
    pub fn discard(self) {
        std::mem::forget(self.guard);
    }

    /// Begin a render pass.
    pub fn with_framebuffer<'a>(
        &mut self,
        framebuffer: &'a Framebuffer,
        clears: &[ClearValue],
    ) -> Result<RenderPassEncoder<'_, 'a>> {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer.begin_render_pass(framebuffer, clears)?;

        Ok(RenderPassEncoder {
            framebuffer,
            render_pass: &framebuffer.info().render_pass,
            inner: &mut self.inner,
        })
    }

    /// Update a buffer with data (directly).
    pub fn update_buffer<T>(&mut self, buffer: &Buffer, offset: u64, data: &[T]) -> Result<()>
    where
        T: bytemuck::Pod,
    {
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: `data` is a slice of `T`, which is `Pod` and `repr(C)`, so it is safe to cast.
        let data = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.command_buffer.update_buffer(buffer, offset, data)
    }

    /// Upload data to a buffer.
    pub fn upload_buffer<T>(
        &mut self,
        buffer: &Buffer,
        offset: u64,
        data: &[T],
        device: &Device,
    ) -> Result<()>
    where
        T: bytemuck::Pod,
    {
        const SMALL_BUFFER_SIZE: u64 = 16384;
        const MIN_ALIGN: usize = 0b11;

        match std::mem::size_of_val(data) as u64 {
            0 => Ok(()),
            size if size <= SMALL_BUFFER_SIZE => self.update_buffer(buffer, offset, data),
            size => {
                let mut staging = device.create_mappable_buffer(
                    BufferInfo {
                        align: MIN_ALIGN.max(std::mem::align_of::<T>() - 1) as u64,
                        size,
                        usage: BufferUsage::TRANSFER_SRC,
                    },
                    gpu_alloc::UsageFlags::UPLOAD | gpu_alloc::UsageFlags::TRANSIENT,
                )?;
                device.upload_to_memory(&mut staging, 0, data)?;

                self.copy_buffer(
                    &staging.freeze(),
                    buffer,
                    &[BufferCopy {
                        src_offset: 0,
                        dst_offset: offset,
                        size,
                    }],
                );
                Ok(())
            }
        }
    }

    /// Copy data between buffer regions.
    pub fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer, regions: &[BufferCopy]) {
        self.command_buffer.copy_buffer(src, dst, regions);
    }

    /// Copy data between images.
    pub fn copy_image(
        &mut self,
        src_image: &Image,
        src_layout: ImageLayout,
        dst_image: &Image,
        dst_layout: ImageLayout,
        regions: &[ImageCopy],
    ) {
        self.command_buffer
            .copy_image(src_image, src_layout, dst_image, dst_layout, regions);
    }

    /// Copy data from a buffer into an image
    pub fn copy_buffer_to_image(
        &mut self,
        src_buffer: &Buffer,
        dst_image: &Image,
        dst_layout: ImageLayout,
        regions: &[BufferImageCopy],
    ) {
        self.command_buffer
            .copy_buffer_to_image(src_buffer, dst_image, dst_layout, regions);
    }

    /// Copy regions of an image, potentially performing format conversion,
    pub fn blit_image(
        &mut self,
        src_image: &Image,
        src_layout: ImageLayout,
        dst_image: &Image,
        dst_layout: ImageLayout,
        regions: &[ImageBlit],
        filter: Filter,
    ) {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer.blit_image(
            src_image, src_layout, dst_image, dst_layout, regions, filter,
        );
    }

    /// Dispatch compute work items.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        assert!(self.capabilities.supports_compute());
        self.command_buffer.dispatch(x, y, z);
    }

    /// Insert a memory dependency.
    pub fn memory_barrier(
        &mut self,
        src: PipelineStageFlags,
        src_access: AccessFlags,
        dst: PipelineStageFlags,
        dst_access: AccessFlags,
    ) {
        self.command_buffer.pipeline_barrier(
            src,
            dst,
            Some(MemoryBarrier {
                src: src_access,
                dst: dst_access,
            }),
            &[],
            &[],
        );
    }

    /// Insert an image memory dependency.
    pub fn image_barriers(
        &mut self,
        src: PipelineStageFlags,
        dst: PipelineStageFlags,
        barriers: &[ImageMemoryBarrier],
    ) {
        self.command_buffer
            .pipeline_barrier(src, dst, None, &[], barriers);
    }

    /// Insert a buffer memory dependency.
    pub fn buffer_barriers(
        &mut self,
        src: PipelineStageFlags,
        dst: PipelineStageFlags,
        barriers: &[BufferMemoryBarrier],
    ) {
        self.command_buffer
            .pipeline_barrier(src, dst, None, barriers, &[]);
    }
}

impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("command_buffer", &self.inner.command_buffer)
            .field("capabilities", &self.inner.capabilities)
            .finish()
    }
}

impl std::ops::Deref for Encoder {
    type Target = EncoderCommon;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for Encoder {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Basic encoder functionality.
pub struct EncoderCommon {
    command_buffer: CommandBuffer,
    capabilities: QueueFlags,
}

impl EncoderCommon {
    /// Set the viewport dynamically for a command buffer.
    pub fn set_viewport(&mut self, viewport: &Viewport) {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer.set_viewport(viewport);
    }

    /// Set the scissor dynamically for a command buffer.
    pub fn set_scissor(&mut self, scissor: &Rect) {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer.set_scissor(scissor);
    }

    /// Bind a graphics pipeline object to a command buffer.
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer.bind_graphics_pipeline(pipeline);
    }

    /// Bind a compute pipeline object to a command buffer.
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        assert!(self.capabilities.supports_compute());
        self.command_buffer.bind_compute_pipeline(pipeline);
    }

    /// Bind vertex buffers to a command buffer starting from the `first_binding`.
    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers: &[(&Buffer, u64)]) {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer
            .bind_vertex_buffers(first_binding, buffers);
    }

    /// Bind an index buffer to a command buffer.
    pub fn bind_index_buffer(&mut self, buffer: &Buffer, offset: u64, index_type: IndexType) {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer
            .bind_index_buffer(buffer, offset, index_type);
    }

    /// Binds descriptor sets for a graphics pipeline to a command buffer.
    pub fn bind_graphics_descriptor_sets(
        &mut self,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        assert!(self.capabilities.supports_graphics());
        self.command_buffer.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            layout,
            first_set,
            descriptor_sets,
            dynamic_offsets,
        )
    }

    /// Binds descriptor sets for a compute pipeline to a command buffer.
    pub fn bind_compute_descriptor_sets(
        &mut self,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        assert!(self.capabilities.supports_compute());
        self.command_buffer.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout,
            first_set,
            descriptor_sets,
            dynamic_offsets,
        )
    }

    /// Update the values of push constants.
    pub fn push_constants<T>(
        &mut self,
        layout: &PipelineLayout,
        stages: ShaderStageFlags,
        offset: u32,
        data: &[T],
    ) where
        T: bytemuck::Pod,
    {
        // SAFETY: `data` is a slice of `T`, which is `Pod` and `repr(C)`, so it is safe to cast.
        let data = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        self.command_buffer
            .push_constants(layout, stages, offset, data);
    }
}

/// Render pass encoder functionality.
pub struct RenderPassEncoder<'a, 'b> {
    framebuffer: &'b Framebuffer,
    render_pass: &'b RenderPass,
    inner: &'a mut EncoderCommon,
}

impl<'a, 'b> RenderPassEncoder<'a, 'b> {
    /// Return the framebuffer associated with this render pass.
    pub fn framebuffer(&self) -> &Framebuffer {
        self.framebuffer
    }

    /// Return the underlying render pass.
    pub fn render_pass(&self) -> &RenderPass {
        self.render_pass
    }

    /// Draw primitives.
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.inner.command_buffer.draw(vertices, instances);
    }

    /// Draw indexed primitives.
    pub fn draw_indexed(&mut self, indices: Range<u32>, vertex_offset: i32, instances: Range<u32>) {
        self.inner
            .command_buffer
            .draw_indexed(indices, vertex_offset, instances);
    }
}

impl std::ops::Deref for RenderPassEncoder<'_, '_> {
    type Target = EncoderCommon;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl std::ops::DerefMut for RenderPassEncoder<'_, '_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

impl Drop for RenderPassEncoder<'_, '_> {
    fn drop(&mut self) {
        self.inner.command_buffer.end_render_pass();
    }
}

struct EncoderDropGuard;

impl Drop for EncoderDropGuard {
    fn drop(&mut self) {
        tracing::error!("encoder must be submitted or discarded before dropping");
    }
}
