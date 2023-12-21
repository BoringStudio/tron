use vulkanalia::vk;

pub use self::buffer::{Buffer, BufferInfo, MappableBuffer};
pub use self::fence::{Fence, FenceState};
pub use self::image::{Image, ImageExtent, ImageInfo, Samples};
pub use self::image_view::{
    ComponentMapping, ImageView, ImageViewInfo, ImageViewType, MakeImageView, Swizzle,
};
pub use self::semaphore::Semaphore;
pub use self::shader_module::{ShaderModule, ShaderModuleInfo};

mod buffer;
mod fence;
mod image;
mod image_view;
mod semaphore;
mod shader_module;

pub trait FormatExt {
    fn aspect_flags(&self) -> vk::ImageAspectFlags;
    fn is_color(&self) -> bool;
    fn is_depth(&self) -> bool;
    fn is_stencil(&self) -> bool;
}

impl FormatExt for vk::Format {
    fn aspect_flags(&self) -> vk::ImageAspectFlags {
        let mut flags = vk::ImageAspectFlags::empty();
        let is_depth = self.is_depth();
        let is_stencil = self.is_stencil();
        if !is_depth && !is_stencil {
            flags |= vk::ImageAspectFlags::COLOR;
        }
        if is_depth {
            flags |= vk::ImageAspectFlags::DEPTH;
        }
        if is_stencil {
            flags |= vk::ImageAspectFlags::STENCIL;
        }
        flags
    }

    fn is_color(&self) -> bool {
        !matches!(
            *self,
            Self::S8_UINT
                | Self::D16_UNORM
                | Self::D16_UNORM_S8_UINT
                | Self::D24_UNORM_S8_UINT
                | Self::D32_SFLOAT
                | Self::D32_SFLOAT_S8_UINT
        )
    }

    fn is_depth(&self) -> bool {
        matches!(
            *self,
            Self::D16_UNORM
                | Self::D16_UNORM_S8_UINT
                | Self::D24_UNORM_S8_UINT
                | Self::D32_SFLOAT
                | Self::D32_SFLOAT_S8_UINT
        )
    }

    fn is_stencil(&self) -> bool {
        matches!(
            *self,
            Self::S8_UINT
                | Self::D16_UNORM_S8_UINT
                | Self::D24_UNORM_S8_UINT
                | Self::D32_SFLOAT_S8_UINT
        )
    }
}
