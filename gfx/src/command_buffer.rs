use vulkanalia::prelude::v1_0::*;

use anyhow::Result;

use crate::device::{Device, WeakDevice};
use crate::queue::QueueId;
use crate::resources::{Buffer, ClearValue, Framebuffer};

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    queue_id: QueueId,
    references: References,
    state: CommandBufferState,
}

impl CommandBuffer {
    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn queue_id(&self) -> QueueId {
        self.queue_id
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

                let info = vk::RenderPassBeginInfo::builder()
                    .render_pass(pass.handle())
                    .framebuffer(framebuffer.handle())
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
            } => todo!(),
        }
    }
}

enum CommandBufferState {
    Full { owner: Device },
    Finished { owned: WeakDevice },
}

#[derive(Debug)]
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
}
