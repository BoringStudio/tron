extern crate self as gfx;

use vulkanalia::vk;

pub use self::device::{CreateRenderPassError, DescriptorAllocError, Device, MapError, WeakDevice};
pub use self::encoder::{
    AccessFlags, BufferCopy, BufferImageCopy, BufferMemoryBarrier, CommandBuffer,
    CommandBufferLevel, Encoder, EncoderCommon, ImageBlit, ImageCopy, ImageLayoutTransition,
    ImageMemoryBarrier, MemoryBarrier, PrimaryEncoder, RenderPassEncoder,
};
pub use self::graphics::{Graphics, InitGraphicsError, InstanceConfig};
pub use self::layout::{AsStd140, AsStd430, Padded, Padding, Std140, Std430};
pub use self::physical::{
    CreateDeviceError, DeviceFeature, DeviceFeatures, DeviceProperties, PhysicalDevice,
};
pub use self::queue::{
    PresentError, PresentStatus, Queue, QueueError, QueueFamily, QueueFlags, QueueId,
    QueueNotFound, QueuesQuery, SingleQueueQuery,
};
pub use self::resources::{
    AttachmentInfo, BlendFactor, BlendOp, Blending, BorderColor, Bounds, Buffer, BufferInfo,
    BufferRange, BufferUsage, BufferView, BufferViewInfo, ClearColor, ClearDepth,
    ClearDepthStencil, ClearValue, ColorBlend, CombinedImageSampler, CompareOp, ComponentMapping,
    ComponentMask, ComputePipeline, ComputePipelineInfo, ComputeShader, CullMode, DepthTest,
    DescriptorBindingFlags, DescriptorSet, DescriptorSetInfo, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorSetLayoutFlags, DescriptorSetLayoutInfo,
    DescriptorSetSize, DescriptorSetWrite, DescriptorSlice, DescriptorType, Fence, FenceState,
    Filter, Format, FormatChannels, FormatDescription, FormatType, FragmentShader, Framebuffer,
    FramebufferInfo, FrontFace, GraphicsPipeline, GraphicsPipelineDescr, GraphicsPipelineInfo,
    GraphicsPipelineRenderingInfo, Image, ImageAspectFlags, ImageExtent, ImageInfo, ImageLayout,
    ImageSubresource, ImageSubresourceLayers, ImageSubresourceRange, ImageUsageFlags, ImageView,
    ImageViewInfo, ImageViewType, IndexType, LoadOp, LogicOp, MakeImageView, MappableBuffer,
    MemoryUsage, MipmapMode, Pipeline, PipelineBindPoint, PipelineLayout, PipelineLayoutInfo,
    PipelineStageFlags, PolygonMode, PrimitiveTopology, PushConstant, Rasterizer, Rect,
    ReductionMode, RenderPass, RenderPassInfo, Sampler, SamplerAddressMode, SamplerInfo, Samples,
    Semaphore, ShaderModule, ShaderModuleInfo, ShaderStageFlags, ShaderType, StencilOp,
    StencilTest, StencilTests, StoreOp, Subpass, SubpassDependency, Swizzle, UpdateDescriptorSet,
    VertexFormat, VertexInputAttribute, VertexInputBinding, VertexInputRate, VertexShader,
    Viewport,
};
pub use self::surface::{
    CreateSurfaceError, PresentMode, Surface, SurfaceError, SurfaceImage, SwapchainSupport,
};
pub use self::types::{DeviceAddress, DeviceLost, OutOfDeviceMemory, State};

pub use gfx_macros::{AsStd140, AsStd430};

mod device;
mod encoder;
mod graphics;
mod layout;
mod physical;
mod queue;
mod resources;
mod surface;
mod types;
mod util;

#[track_caller]
pub(crate) fn out_of_host_memory() -> ! {
    std::alloc::handle_alloc_error(unsafe { std::alloc::Layout::from_size_align_unchecked(1, 1) })
}

#[track_caller]
#[cold]
pub(crate) fn unexpected_vulkan_error(e: vk::ErrorCode) -> ! {
    panic!("unexpected Vulkan error: {e}")
}

#[doc(hidden)]
pub mod inner_proc_stuff {
    pub use bytemuck;

    /// Gives the number of bytes needed to make `offset` be aligned to `align_mask`.
    pub const fn align_offset(align_mask: u64, offset: usize) -> u64 {
        let offset = offset as u64;
        if offset & align_mask == 0 {
            0
        } else {
            align_mask + 1 - (offset & align_mask)
        }
    }
}
