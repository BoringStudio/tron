use std::mem::ManuallyDrop;
use std::num::NonZeroU64;
use std::sync::Arc;

use glam::{UVec2, UVec3};
use gpu_alloc::MemoryBlock;
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::util::{FromGfx, ToVk};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImageExtent {
    D1 { width: u32 },
    D2 { width: u32, height: u32 },
    D3 { width: u32, height: u32, depth: u32 },
}

impl From<u32> for ImageExtent {
    fn from(value: u32) -> Self {
        Self::D1 { width: value }
    }
}

impl From<vk::Extent2D> for ImageExtent {
    fn from(value: vk::Extent2D) -> Self {
        Self::D2 {
            width: value.width,
            height: value.height,
        }
    }
}

impl From<UVec2> for ImageExtent {
    fn from(value: UVec2) -> Self {
        Self::D2 {
            width: value.x,
            height: value.y,
        }
    }
}

impl From<vk::Extent3D> for ImageExtent {
    fn from(value: vk::Extent3D) -> Self {
        Self::D3 {
            width: value.width,
            height: value.height,
            depth: value.depth,
        }
    }
}

impl From<UVec3> for ImageExtent {
    fn from(value: UVec3) -> Self {
        Self::D3 {
            width: value.x,
            height: value.y,
            depth: value.z,
        }
    }
}

impl FromGfx<ImageExtent> for vk::Extent2D {
    fn from_gfx(value: ImageExtent) -> Self {
        let e = vk::Extent2D::builder();
        match value {
            ImageExtent::D1 { width } => e.width(width),
            ImageExtent::D2 { width, height } => e.width(width).height(height),
            ImageExtent::D3 { width, height, .. } => e.width(width).height(height),
        }
        .build()
    }
}

impl From<ImageExtent> for UVec2 {
    fn from(value: ImageExtent) -> Self {
        match value {
            ImageExtent::D1 { width } => UVec2::new(width, 0),
            ImageExtent::D2 { width, height } => UVec2::new(width, height),
            ImageExtent::D3 { width, height, .. } => UVec2::new(width, height),
        }
    }
}

impl FromGfx<ImageExtent> for vk::Extent3D {
    fn from_gfx(value: ImageExtent) -> Self {
        let e = vk::Extent3D::builder();
        match value {
            ImageExtent::D1 { width } => e.width(width),
            ImageExtent::D2 { width, height } => e.width(width).height(height),
            ImageExtent::D3 {
                width,
                height,
                depth,
            } => e.width(width).height(height).depth(depth),
        }
        .build()
    }
}

impl From<ImageExtent> for UVec3 {
    fn from(value: ImageExtent) -> Self {
        match value {
            ImageExtent::D1 { width } => UVec3::new(width, 0, 0),
            ImageExtent::D2 { width, height } => UVec3::new(width, height, 0),
            ImageExtent::D3 {
                width,
                height,
                depth,
            } => UVec3::new(width, height, depth),
        }
    }
}

impl FromGfx<ImageExtent> for vk::ImageType {
    fn from_gfx(value: ImageExtent) -> Self {
        match value {
            ImageExtent::D1 { .. } => Self::_1D,
            ImageExtent::D2 { .. } => Self::_2D,
            ImageExtent::D3 { .. } => Self::_3D,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Samples {
    _1,
    _2,
    _4,
    _8,
    _16,
    _32,
    _64,
}

impl FromGfx<Samples> for vk::SampleCountFlags {
    fn from_gfx(value: Samples) -> Self {
        match value {
            Samples::_1 => vk::SampleCountFlags::_1,
            Samples::_2 => vk::SampleCountFlags::_2,
            Samples::_4 => vk::SampleCountFlags::_4,
            Samples::_8 => vk::SampleCountFlags::_8,
            Samples::_16 => vk::SampleCountFlags::_16,
            Samples::_32 => vk::SampleCountFlags::_32,
            Samples::_64 => vk::SampleCountFlags::_64,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageLayout {
    General,
    ColorAttachmentOptimal,
    DepthStencilAttachmentOptimal,
    DepthStencilReadOnlyOptimal,
    ShaderReadOnlyOptimal,
    TransferSrcOptimal,
    TransferDstOptimal,
    Present,
}

impl FromGfx<ImageLayout> for vk::ImageLayout {
    fn from_gfx(value: ImageLayout) -> Self {
        match value {
            ImageLayout::General => Self::GENERAL,
            ImageLayout::ColorAttachmentOptimal => Self::COLOR_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilAttachmentOptimal => Self::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilReadOnlyOptimal => Self::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ImageLayout::ShaderReadOnlyOptimal => Self::SHADER_READ_ONLY_OPTIMAL,
            ImageLayout::TransferSrcOptimal => Self::TRANSFER_SRC_OPTIMAL,
            ImageLayout::TransferDstOptimal => Self::TRANSFER_DST_OPTIMAL,
            ImageLayout::Present => Self::PRESENT_SRC_KHR,
        }
    }
}

impl FromGfx<Option<ImageLayout>> for vk::ImageLayout {
    #[inline]
    fn from_gfx(value: Option<ImageLayout>) -> Self {
        match value {
            Some(value) => value.to_vk(),
            None => Self::UNDEFINED,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageInfo {
    pub extent: ImageExtent,
    pub format: Format,
    pub mip_levels: u32,
    pub samples: Samples,
    pub array_layers: u32,
    pub usage: ImageUsageFlags,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct ImageUsageFlags: u32 {
        const TRANSFER_SRC = 1;
        const TRANSFER_DST = 1 << 1;
        const SAMPLED = 1 << 2;
        const STORAGE = 1 << 3;
        const COLOR_ATTACHMENT = 1 << 4;
        const DEPTH_STENCIL_ATTACHMENT = 1 << 5;
        const INPUT_ATTACHMENT = 1 << 7;
    }
}

impl FromGfx<ImageUsageFlags> for vk::ImageUsageFlags {
    fn from_gfx(value: ImageUsageFlags) -> Self {
        let mut res = Self::empty();
        if value.contains(ImageUsageFlags::TRANSFER_SRC) {
            res |= Self::TRANSFER_SRC;
        }
        if value.contains(ImageUsageFlags::TRANSFER_DST) {
            res |= Self::TRANSFER_DST;
        }
        if value.contains(ImageUsageFlags::SAMPLED) {
            res |= Self::SAMPLED;
        }
        if value.contains(ImageUsageFlags::STORAGE) {
            res |= Self::STORAGE;
        }
        if value.contains(ImageUsageFlags::COLOR_ATTACHMENT) {
            res |= Self::COLOR_ATTACHMENT;
        }
        if value.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            res |= Self::DEPTH_STENCIL_ATTACHMENT;
        }
        if value.contains(ImageUsageFlags::INPUT_ATTACHMENT) {
            res |= Self::INPUT_ATTACHMENT;
        }
        res
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct Image {
    inner: Arc<Inner>,
}

impl Image {
    pub(crate) fn new(
        handle: vk::Image,
        info: ImageInfo,
        owner: WeakDevice,
        block: gpu_alloc::MemoryBlock<vk::DeviceMemory>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
                source: ImageSource::Device {
                    memory_block: ManuallyDrop::new(block),
                },
            }),
        }
    }

    pub(crate) fn new_surface(
        handle: vk::Image,
        info: ImageInfo,
        owner: WeakDevice,
        id: NonZeroU64,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
                source: ImageSource::Surface { id },
            }),
        }
    }

    pub fn info(&self) -> &ImageInfo {
        &self.inner.info
    }

    pub fn handle(&self) -> vk::Image {
        self.inner.handle
    }

    pub fn try_dispose_as_surface(mut self) -> Result<(), Self> {
        if matches!(&self.inner.source, ImageSource::Surface { .. })
            && Arc::get_mut(&mut self.inner).is_some()
        {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("Image")
                .field("info", &self.inner.info)
                .field("owner", &self.inner.owner)
                .field("handle", &self.inner.handle)
                .field("source", &self.inner.source)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for Image {}
impl PartialEq for Image {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for Image {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::Image,
    info: ImageInfo,
    source: ImageSource,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let ImageSource::Device { memory_block } = &mut self.source else {
            // NOTE: surface images are destroyed externally
            return;
        };

        unsafe {
            let block = ManuallyDrop::take(memory_block);

            if let Some(device) = self.owner.upgrade() {
                device.destroy_image(self.handle, block);
            }

            // NOTE: `Relevant` will preintln error here if device was already destroyed
        }
    }
}

enum ImageSource {
    Device {
        memory_block: ManuallyDrop<MemoryBlock<vk::DeviceMemory>>,
    },
    Surface {
        id: NonZeroU64,
    },
}

impl std::fmt::Debug for ImageSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Device { memory_block } => f
                .debug_struct("ImageSource::Device")
                .field("memory_handle", memory_block.memory())
                .field("memory_offset", &memory_block.offset())
                .field("memory_size", &memory_block.size())
                .finish(),
            Self::Surface { id } => f
                .debug_struct("ImageSource::Surface")
                .field("id", &id.get())
                .finish(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum FormatChannels {
    R,
    RG,
    RGB,
    BGR,
    RGBA,
    BGRA,
    D,
    S,
    DS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum FormatType {
    Uint,
    Sint,
    Srgb,
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Sfloat,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct FormatDescription<Channels, Bits, Type> {
    pub channels: Channels,
    pub bits: Bits,
    pub ty: Type,
}

macro_rules! declare_format {
    (
        $enum_name:ident,
        {
            $($ident:ident => $orig:ident as ($channels:ident, $bits:literal, $ty:ident)),*$(,)?
        }) => {
        #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
        pub enum $enum_name {
            $($ident),*,
        }

        impl $enum_name {
            pub fn description(&self) -> FormatDescription<FormatChannels, u32, FormatType> {
                match self {
                    $(Self::$ident => FormatDescription {
                        channels: FormatChannels::$channels,
                        bits: $bits,
                        ty: FormatType::$ty,
                    }),*,
                }
            }

            pub fn from_vk(format: vk::Format) -> Option<Self> {
                match format {
                    $(vk::Format::$orig => Some($enum_name::$ident)),*,
                    _ => None,
                }
            }
        }

        impl FromGfx<$enum_name> for vk::Format {
            fn from_gfx(value: $enum_name) -> Self {
                match value {
                    $($enum_name::$ident => vk::Format::$orig),*,
                }
            }
        }
    };
}

declare_format! {
    Format, {
        R8Unorm => R8_UNORM as (R, 8, Unorm),
        R8Snorm => R8_SNORM as (R, 8, Snorm),
        R8Uscaled => R8_USCALED as (R, 8, Uscaled),
        R8Sscaled => R8_SSCALED as (R, 8, Sscaled),
        R8Uint => R8_UINT as (R, 8, Uint),
        R8Sint => R8_SINT as (R, 8, Sint),
        R8Srgb => R8_SRGB as (R, 8, Srgb),

        RG8Unorm => R8G8_UNORM as (RG, 8, Unorm),
        RG8Snorm => R8G8_SNORM as (RG, 8, Snorm),
        RG8Uscaled => R8G8_USCALED as (RG, 8, Uscaled),
        RG8Sscaled => R8G8_SSCALED as (RG, 8, Sscaled),
        RG8Uint => R8G8_UINT as (RG, 8, Uint),
        RG8Sint => R8G8_SINT as (RG, 8, Sint),
        RG8Srgb => R8G8_SRGB as (RG, 8, Srgb),

        RGB8Unorm => R8G8B8_UNORM as (RGB, 8, Unorm),
        RGB8Snorm => R8G8B8_SNORM as (RGB, 8, Snorm),
        RGB8Uscaled => R8G8B8_USCALED as (RGB, 8, Uscaled),
        RGB8Sscaled => R8G8B8_SSCALED as (RGB, 8, Sscaled),
        RGB8Uint => R8G8B8_UINT as (RGB, 8, Uint),
        RGB8Sint => R8G8B8_SINT as (RGB, 8, Sint),
        RGB8Srgb => R8G8B8_SRGB as (RGB, 8, Srgb),

        BGR8Unorm => B8G8R8_UNORM as (BGR, 8, Unorm),
        BGR8Snorm => B8G8R8_SNORM as (BGR, 8, Snorm),
        BGR8Uscaled => B8G8R8_USCALED as (BGR, 8, Uscaled),
        BGR8Sscaled => B8G8R8_SSCALED as (BGR, 8, Sscaled),
        BGR8Uint => B8G8R8_UINT as (BGR, 8, Uint),
        BGR8Sint => B8G8R8_SINT as (BGR, 8, Sint),
        BGR8Srgb => B8G8R8_SRGB as (BGR, 8, Srgb),

        RGBA8Unorm => R8G8B8A8_UNORM as (RGBA, 8, Unorm),
        RGBA8Snorm => R8G8B8A8_SNORM as (RGBA, 8, Snorm),
        RGBA8Uscaled => R8G8B8A8_USCALED as (RGBA, 8, Uscaled),
        RGBA8Sscaled => R8G8B8A8_SSCALED as (RGBA, 8, Sscaled),
        RGBA8Uint => R8G8B8A8_UINT as (RGBA, 8, Uint),
        RGBA8Sint => R8G8B8A8_SINT as (RGBA, 8, Sint),
        RGBA8Srgb => R8G8B8A8_SRGB as (RGBA, 8, Srgb),

        BGRA8Unorm => B8G8R8A8_UNORM as (BGRA, 8, Unorm),
        BGRA8Snorm => B8G8R8A8_SNORM as (BGRA, 8, Snorm),
        BGRA8Uscaled => B8G8R8A8_USCALED as (BGRA, 8, Uscaled),
        BGRA8Sscaled => B8G8R8A8_SSCALED as (BGRA, 8, Sscaled),
        BGRA8Uint => B8G8R8A8_UINT as (BGRA, 8, Uint),
        BGRA8Sint => B8G8R8A8_SINT as (BGRA, 8, Sint),
        BGRA8Srgb => B8G8R8A8_SRGB as (BGRA, 8, Srgb),

        R16Unorm => R16_UNORM as (R, 16, Unorm),
        R16Snorm => R16_SNORM as (R, 16, Snorm),
        R16Uscaled => R16_USCALED as (R, 16, Uscaled),
        R16Sscaled => R16_SSCALED as (R, 16, Sscaled),
        R16Uint => R16_UINT as (R, 16, Uint),
        R16Sint => R16_SINT as (R, 16, Sint),
        R16Sfloat => R16_SFLOAT as (R, 16, Sfloat),

        RG16Unorm => R16G16_UNORM as (RG, 16, Unorm),
        RG16Snorm => R16G16_SNORM as (RG, 16, Snorm),
        RG16Uscaled => R16G16_USCALED as (RG, 16, Uscaled),
        RG16Sscaled => R16G16_SSCALED as (RG, 16, Sscaled),
        RG16Uint => R16G16_UINT as (RG, 16, Uint),
        RG16Sint => R16G16_SINT as (RG, 16, Sint),
        RG16Sfloat => R16G16_SFLOAT as (RG, 16, Sfloat),

        RGB16Unorm => R16G16B16_UNORM as (RGB, 16, Unorm),
        RGB16Snorm => R16G16B16_SNORM as (RGB, 16, Snorm),
        RGB16Uscaled => R16G16B16_USCALED as (RGB, 16, Uscaled),
        RGB16Sscaled => R16G16B16_SSCALED as (RGB, 16, Sscaled),
        RGB16Uint => R16G16B16_UINT as (RGB, 16, Uint),
        RGB16Sint => R16G16B16_SINT as (RGB, 16, Sint),
        RGB16Sfloat => R16G16B16_SFLOAT as (RGB, 16, Sfloat),

        RGBA16Unorm => R16G16B16A16_UNORM as (RGBA, 16, Unorm),
        RGBA16Snorm => R16G16B16A16_SNORM as (RGBA, 16, Snorm),
        RGBA16Uscaled => R16G16B16A16_USCALED as (RGBA, 16, Uscaled),
        RGBA16Sscaled => R16G16B16A16_SSCALED as (RGBA, 16, Sscaled),
        RGBA16Uint => R16G16B16A16_UINT as (RGBA, 16, Uint),
        RGBA16Sint => R16G16B16A16_SINT as (RGBA, 16, Sint),
        RGBA16Sfloat => R16G16B16A16_SFLOAT as (RGBA, 16, Sfloat),

        R32Uint => R32_UINT as (R, 32, Uint),
        R32Sint => R32_SINT as (R, 32, Sint),
        R32Sfloat => R32_SFLOAT as (R, 32, Sfloat),

        RG32Uint => R32G32_UINT as (RG, 32, Uint),
        RG32Sint => R32G32_SINT as (RG, 32, Sint),
        RG32Sfloat => R32G32_SFLOAT as (RG, 32, Sfloat),

        RGB32Uint => R32G32B32_UINT as (RGB, 32, Uint),
        RGB32Sint => R32G32B32_SINT as (RGB, 32, Sint),
        RGB32Sfloat => R32G32B32_SFLOAT as (RGB, 32, Sfloat),

        RGBA32Uint => R32G32B32A32_UINT as (RGBA, 32, Uint),
        RGBA32Sint => R32G32B32A32_SINT as (RGBA, 32, Sint),
        RGBA32Sfloat => R32G32B32A32_SFLOAT as (RGBA, 32, Sfloat),

        R64Uint => R64_UINT as (R, 64, Uint),
        R64Sint => R64_SINT as (R, 64, Sint),
        R64Sfloat => R64_SFLOAT as (R, 64, Sfloat),

        RG64Uint => R64G64_UINT as (RG, 64, Uint),
        RG64Sint => R64G64_SINT as (RG, 64, Sint),
        RG64Sfloat => R64G64_SFLOAT as (RG, 64, Sfloat),

        RGB64Uint => R64G64B64_UINT as (RGB, 64, Uint),
        RGB64Sint => R64G64B64_SINT as (RGB, 64, Sint),
        RGB64Sfloat => R64G64B64_SFLOAT as (RGB, 64, Sfloat),

        RGBA64Uint => R64G64B64A64_UINT as (RGBA, 64, Uint),
        RGBA64Sint => R64G64B64A64_SINT as (RGBA, 64, Sint),
        RGBA64Sfloat => R64G64B64A64_SFLOAT as (RGBA, 64, Sfloat),

        D16Unorm => D16_UNORM as (D, 16, Unorm),
        D32Sfloat => D32_SFLOAT as (D, 32, Sfloat),
        S8Uint => S8_UINT as (S, 8, Uint),
        D16UnormS8Uint => D16_UNORM_S8_UINT as (DS, 16, Unorm),
        D24UnormS8Uint => D24_UNORM_S8_UINT as (DS, 24, Unorm),
        D32SfloatS8Uint => D32_SFLOAT_S8_UINT as (DS, 32, Sfloat),
    }
}

impl Format {
    pub fn aspect_flags(&self) -> ImageAspectFlags {
        let mut flags = ImageAspectFlags::empty();
        let is_depth = self.is_depth();
        let is_stencil = self.is_stencil();
        if !is_depth && !is_stencil {
            flags |= ImageAspectFlags::COLOR;
        }
        if is_depth {
            flags |= ImageAspectFlags::DEPTH;
        }
        if is_stencil {
            flags |= ImageAspectFlags::STENCIL;
        }
        flags
    }

    pub fn is_color(&self) -> bool {
        !matches!(
            *self,
            Self::S8Uint
                | Self::D16Unorm
                | Self::D16UnormS8Uint
                | Self::D24UnormS8Uint
                | Self::D32Sfloat
                | Self::D32SfloatS8Uint
        )
    }

    pub fn is_depth(&self) -> bool {
        matches!(
            *self,
            Self::D16Unorm
                | Self::D16UnormS8Uint
                | Self::D24UnormS8Uint
                | Self::D32Sfloat
                | Self::D32SfloatS8Uint
        )
    }

    pub fn is_stencil(&self) -> bool {
        matches!(
            *self,
            Self::S8Uint | Self::D16UnormS8Uint | Self::D24UnormS8Uint | Self::D32SfloatS8Uint
        )
    }
}

impl FromGfx<Option<Format>> for vk::Format {
    #[inline]
    fn from_gfx(value: Option<Format>) -> Self {
        match value {
            Some(value) => value.to_vk(),
            None => Self::UNDEFINED,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct ImageAspectFlags: u8 {
        const COLOR = 1;
        const DEPTH = 1 << 1;
        const STENCIL = 1 << 2;
    }
}

impl FromGfx<ImageAspectFlags> for vk::ImageAspectFlags {
    fn from_gfx(value: ImageAspectFlags) -> Self {
        let mut res = Self::empty();
        if value.contains(ImageAspectFlags::COLOR) {
            res |= Self::COLOR;
        }
        if value.contains(ImageAspectFlags::DEPTH) {
            res |= Self::DEPTH;
        }
        if value.contains(ImageAspectFlags::STENCIL) {
            res |= Self::STENCIL;
        }
        res
    }
}
