use vulkanalia::prelude::v1_0::*;

use anyhow::Result;
use bumpalo::Bump;

use crate::device::{Device, WeakDevice};
use crate::queue::QueueId;
use crate::resources::{Buffer, ClearValue, Framebuffer, IndexType, LoadOp};

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

    pub fn write(&mut self, command: Command<'_>) -> Result<()> {
        let device = match &self.state {
            CommandBufferState::Full { owner } => owner,
            CommandBufferState::Finished { .. } => return Ok(()),
        };
        let logical = device.logical();

        let references = &mut self.references;

        match command {
            Command::BeginRenderPass { framebuffer, clear } => {
                references.framebuffers.push(framebuffer.clone());

                let pass = &framebuffer.info().render_pass;

                let mut clear = clear.into_iter();
                let mut clear_values_invalid = false;
                let clear_values =
                    self.alloc
                        .alloc_slice_fill_iter(pass.info().attachments.iter().map(|attachment| {
                            if attachment.load_op == LoadOp::Clear(()) {
                                if let Some(clear) =
                                    clear.next().and_then(|v| v.to_vk(attachment.format))
                                {
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
                        extent: framebuffer.info().extent,
                    });

                Ok(unsafe {
                    logical.cmd_begin_render_pass(self.handle, &info, vk::SubpassContents::INLINE)
                })
            }
            Command::EndRenderPass => Ok(unsafe { logical.cmd_end_render_pass(self.handle) }),
            Command::SetViewport { ref viewport } => Ok(unsafe {
                logical.cmd_set_viewport(self.handle, 0, std::slice::from_ref(viewport))
            }),
            Command::SetScissor { ref scissors } => Ok(unsafe {
                logical.cmd_set_scissor(self.handle, 0, std::slice::from_ref(scissors))
            }),
            Command::Draw {
                vertices,
                instances,
            } => Ok(unsafe {
                logical.cmd_draw(
                    self.handle,
                    vertices.end - vertices.start,
                    instances.end - instances.start,
                    vertices.start,
                    instances.start,
                )
            }),
            Command::DrawIndexed {
                indices,
                vertex_offset,
                instances,
            } => Ok(unsafe {
                logical.cmd_draw_indexed(
                    self.handle,
                    indices.end - indices.start,
                    instances.end - instances.start,
                    indices.start,
                    vertex_offset,
                    instances.start,
                )
            }),
            Command::UpdateBuffer {
                buffer,
                offset,
                data,
            } => {
                anyhow::ensure!(offset % 4 == 0, "unaligned buffer offset");
                anyhow::ensure!(data.len() % 4 == 0, "unaligned buffer data length");
                anyhow::ensure!(data.len() <= 65536, "too much data to update");

                Ok(
                    unsafe {
                        logical.cmd_update_buffer(self.handle, buffer.handle(), offset, data)
                    },
                )
            }
            Command::BindVertexBuffers { first, buffers } => {
                for &(buffer, _) in buffers {
                    references.buffers.push(buffer.clone());
                }

                let offsets = self
                    .alloc
                    .alloc_slice_fill_iter(buffers.iter().map(|&(_, offset)| offset));
                let buffers = self
                    .alloc
                    .alloc_slice_fill_iter(buffers.iter().map(|&(buffer, _)| buffer.handle()));

                unsafe { logical.cmd_bind_vertex_buffers(self.handle, first, buffers, offsets) };

                self.alloc.reset();
                Ok(())
            }
            Command::BindIndexBuffer {
                buffer,
                offset,
                index_type,
            } => {
                references.buffers.push(buffer.clone());

                Ok(unsafe {
                    logical.cmd_bind_index_buffer(
                        self.handle,
                        buffer.handle(),
                        offset,
                        index_type.into(),
                    )
                })
            }
        }
    }
}

enum CommandBufferState {
    Full { owner: Device },
    Finished { owner: WeakDevice },
}

#[derive(Default, Debug)]
pub struct References {
    buffers: Vec<Buffer>,
    framebuffers: Vec<Framebuffer>,
}

impl References {
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.framebuffers.clear();
    }
}

pub enum Command<'a> {
    BeginRenderPass {
        framebuffer: &'a Framebuffer,
        clear: &'a [ClearValue],
    },
    EndRenderPass,

    SetViewport {
        viewport: vk::Viewport,
    },
    SetScissor {
        scissors: vk::Rect2D,
    },

    Draw {
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    },
    DrawIndexed {
        indices: std::ops::Range<u32>,
        vertex_offset: i32,
        instances: std::ops::Range<u32>,
    },

    UpdateBuffer {
        buffer: &'a Buffer,
        offset: u64,
        data: &'a [u8],
    },
    BindVertexBuffers {
        first: u32,
        buffers: &'a [(&'a Buffer, u64)],
    },
    BindIndexBuffer {
        buffer: &'a Buffer,
        offset: u64,
        index_type: IndexType,
    },
}
