use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::device::{Device, WeakDevice};
use crate::resources::{Image, ImageAspectFlags, ImageExtent, ImageInfo};
use crate::util::{FromGfx, ToVk};

/// An object that can be used to create an [`ImageView`].
pub trait MakeImageView {
    fn make_image_view(&self, device: &Device) -> Result<ImageView>;
}

impl MakeImageView for Image {
    fn make_image_view(&self, device: &Device) -> Result<ImageView> {
        let info = ImageViewInfo::new(self.clone());
        device.create_image_view(info)
    }
}

impl MakeImageView for ImageView {
    fn make_image_view(&self, _device: &Device) -> Result<ImageView> {
        Ok(self.clone())
    }
}

/// Image view dimensions.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageViewType {
    D1,
    D2,
    D3,
    Cube,
}

impl FromGfx<ImageViewType> for vk::ImageViewType {
    fn from_gfx(value: ImageViewType) -> Self {
        match value {
            ImageViewType::D1 => vk::ImageViewType::_1D,
            ImageViewType::D2 => vk::ImageViewType::_2D,
            ImageViewType::D3 => vk::ImageViewType::_3D,
            ImageViewType::Cube => vk::ImageViewType::CUBE,
        }
    }
}

/// Specify how a component is swizzled.
#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Swizzle {
    /// Component is set to the identity swizzle.
    #[default]
    Identity,
    /// Component is set to zero.
    Zero,
    /// Component is set to one.
    One,
    /// Component is set to the value of the R component of the image.
    R,
    /// Component is set to the value of the G component of the image.
    G,
    /// Component is set to the value of the B component of the image.
    B,
    /// Component is set to the value of the A component of the image.
    A,
}

impl FromGfx<Swizzle> for vk::ComponentSwizzle {
    fn from_gfx(value: Swizzle) -> Self {
        match value {
            Swizzle::Identity => vk::ComponentSwizzle::IDENTITY,
            Swizzle::Zero => vk::ComponentSwizzle::ZERO,
            Swizzle::One => vk::ComponentSwizzle::ONE,
            Swizzle::R => vk::ComponentSwizzle::R,
            Swizzle::G => vk::ComponentSwizzle::G,
            Swizzle::B => vk::ComponentSwizzle::B,
            Swizzle::A => vk::ComponentSwizzle::A,
        }
    }
}

/// Structure specifying a color component mapping.
#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ComponentMapping {
    pub r: Swizzle,
    pub g: Swizzle,
    pub b: Swizzle,
    pub a: Swizzle,
}

impl FromGfx<ComponentMapping> for vk::ComponentMapping {
    fn from_gfx(value: ComponentMapping) -> Self {
        Self {
            r: value.r.to_vk(),
            g: value.g.to_vk(),
            b: value.b.to_vk(),
            a: value.a.to_vk(),
        }
    }
}

/// Structure specifying parameters of a newly created image view
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewInfo {
    pub ty: ImageViewType,
    pub range: ImageSubresourceRange,
    pub image: Image,
    pub mapping: ComponentMapping,
}

impl ImageViewInfo {
    pub fn new(image: Image) -> Self {
        let image_info = image.info();

        Self {
            ty: match &image_info.extent {
                ImageExtent::D1 { .. } => ImageViewType::D1,
                ImageExtent::D2 { .. } => ImageViewType::D2,
                ImageExtent::D3 { .. } => ImageViewType::D3,
            },
            range: ImageSubresourceRange::whole(image_info),
            image,
            mapping: Default::default(),
        }
    }

    pub fn is_whole_image(&self, image: &Image) -> bool {
        self.image == *image
            && self.range == ImageSubresourceRange::whole(image.info())
            && self.mapping == ComponentMapping::default()
            && matches!(
                (self.ty, &image.info().extent),
                (ImageViewType::D1, ImageExtent::D1 { .. })
                    | (ImageViewType::D2, ImageExtent::D2 { .. })
                    | (ImageViewType::D3, ImageExtent::D3 { .. })
            )
    }
}

/// Structure specifying an image subresource range.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageSubresourceRange {
    pub aspect: ImageAspectFlags,
    pub first_mip_level: u32,
    pub mip_level_count: u32,
    pub first_array_layer: u32,
    pub array_layer_count: u32,
}

impl ImageSubresourceRange {
    pub fn new(aspect: ImageAspectFlags, mip_levels: Range<u32>, array_layers: Range<u32>) -> Self {
        Self {
            aspect,
            first_mip_level: mip_levels.start,
            mip_level_count: mip_levels.end - mip_levels.start,
            first_array_layer: array_layers.start,
            array_layer_count: array_layers.end - array_layers.start,
        }
    }

    pub fn whole(info: &ImageInfo) -> Self {
        Self {
            aspect: info.format.aspect_flags(),
            first_mip_level: 0,
            mip_level_count: info.mip_levels,
            first_array_layer: 0,
            array_layer_count: info.array_layers,
        }
    }

    pub fn color(mip_levels: Range<u32>, array_layers: Range<u32>) -> Self {
        Self::new(ImageAspectFlags::COLOR, mip_levels, array_layers)
    }

    pub fn depth(mip_levels: Range<u32>, array_layers: Range<u32>) -> Self {
        Self::new(ImageAspectFlags::DEPTH, mip_levels, array_layers)
    }

    pub fn stencil(mip_levels: Range<u32>, array_layers: Range<u32>) -> Self {
        Self::new(ImageAspectFlags::STENCIL, mip_levels, array_layers)
    }

    pub fn depth_stencil(mip_levels: Range<u32>, array_layers: Range<u32>) -> Self {
        Self::new(
            ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
            mip_levels,
            array_layers,
        )
    }
}

impl FromGfx<ImageSubresourceRange> for vk::ImageSubresourceRange {
    fn from_gfx(value: ImageSubresourceRange) -> Self {
        vk::ImageSubresourceRange::builder()
            .aspect_mask(value.aspect.to_vk())
            .base_mip_level(value.first_mip_level)
            .level_count(value.mip_level_count)
            .base_array_layer(value.first_array_layer)
            .layer_count(value.array_layer_count)
            .build()
    }
}

impl From<ImageSubresourceLayers> for ImageSubresourceRange {
    fn from(value: ImageSubresourceLayers) -> Self {
        Self {
            aspect: value.aspect,
            first_mip_level: value.mip_level,
            mip_level_count: 1,
            first_array_layer: value.first_array_layer,
            array_layer_count: value.array_layer_count,
        }
    }
}

impl From<ImageSubresource> for ImageSubresourceRange {
    fn from(value: ImageSubresource) -> Self {
        Self {
            aspect: value.aspect,
            first_mip_level: value.mip_level,
            mip_level_count: 1,
            first_array_layer: value.array_layer,
            array_layer_count: 1,
        }
    }
}

/// Structure specifying an image subresource layers.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageSubresourceLayers {
    pub aspect: ImageAspectFlags,
    pub mip_level: u32,
    pub first_array_layer: u32,
    pub array_layer_count: u32,
}

impl ImageSubresourceLayers {
    pub fn new(aspect: ImageAspectFlags, mip_level: u32, array_layers: Range<u32>) -> Self {
        Self {
            aspect,
            mip_level,
            first_array_layer: array_layers.start,
            array_layer_count: array_layers.end - array_layers.start,
        }
    }

    pub fn all_layers(info: &ImageInfo, mip_level: u32) -> Self {
        Self {
            aspect: info.format.aspect_flags(),
            mip_level,
            first_array_layer: 0,
            array_layer_count: info.array_layers,
        }
    }

    pub fn color(mip_level: u32, array_layers: Range<u32>) -> Self {
        Self::new(ImageAspectFlags::COLOR, mip_level, array_layers)
    }

    pub fn depth(mip_level: u32, array_layers: Range<u32>) -> Self {
        Self::new(ImageAspectFlags::DEPTH, mip_level, array_layers)
    }

    pub fn stencil(mip_level: u32, array_layers: Range<u32>) -> Self {
        Self::new(ImageAspectFlags::STENCIL, mip_level, array_layers)
    }

    pub fn depth_stencil(mip_level: u32, array_layers: Range<u32>) -> Self {
        Self::new(
            ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
            mip_level,
            array_layers,
        )
    }
}

impl FromGfx<ImageSubresourceLayers> for vk::ImageSubresourceLayers {
    fn from_gfx(value: ImageSubresourceLayers) -> Self {
        Self::builder()
            .aspect_mask(value.aspect.to_vk())
            .mip_level(value.mip_level)
            .base_array_layer(value.first_array_layer)
            .layer_count(value.array_layer_count)
            .build()
    }
}

impl From<ImageSubresource> for ImageSubresourceLayers {
    fn from(value: ImageSubresource) -> Self {
        Self {
            aspect: value.aspect,
            mip_level: value.mip_level,
            first_array_layer: value.array_layer,
            array_layer_count: 1,
        }
    }
}

/// Structure specifying an image subresource.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ImageSubresource {
    pub aspect: ImageAspectFlags,
    pub mip_level: u32,
    pub array_layer: u32,
}

impl ImageSubresource {
    pub fn new(aspect: ImageAspectFlags, mip_level: u32, array_layer: u32) -> Self {
        Self {
            aspect,
            mip_level,
            array_layer,
        }
    }

    pub fn from_info(info: &ImageInfo, mip_level: u32, array_layer: u32) -> Self {
        Self {
            aspect: info.format.aspect_flags(),
            mip_level,
            array_layer,
        }
    }

    pub fn color(mip_level: u32, array_layer: u32) -> Self {
        Self::new(ImageAspectFlags::COLOR, mip_level, array_layer)
    }

    pub fn depth(mip_level: u32, array_layer: u32) -> Self {
        Self::new(ImageAspectFlags::DEPTH, mip_level, array_layer)
    }

    pub fn stencil(mip_level: u32, array_layer: u32) -> Self {
        Self::new(ImageAspectFlags::STENCIL, mip_level, array_layer)
    }

    pub fn depth_stencil(mip_level: u32, array_layer: u32) -> Self {
        Self::new(
            ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
            mip_level,
            array_layer,
        )
    }
}

impl FromGfx<ImageSubresource> for vk::ImageSubresource {
    fn from_gfx(value: ImageSubresource) -> Self {
        Self::builder()
            .aspect_mask(value.aspect.to_vk())
            .mip_level(value.mip_level)
            .array_layer(value.array_layer)
            .build()
    }
}

/// A wrapper around a Vulkan image view object.
///
/// Image objects are not directly accessed by pipeline shaders for reading or
/// writing image data. Instead, image views representing contiguous ranges of
/// the image subresources and containing additional metadata are used for that
/// purpose. Views must be created on images of compatible types, and must represent
/// a valid subset of image subresources.
#[derive(Clone)]
#[repr(transparent)]
pub struct ImageView {
    inner: Arc<Inner>,
}

impl ImageView {
    pub(crate) fn new(handle: vk::ImageView, info: ImageViewInfo, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn info(&self) -> &ImageViewInfo {
        &self.inner.info
    }

    pub fn handle(&self) -> vk::ImageView {
        self.inner.handle
    }
}

impl std::fmt::Debug for ImageView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("ImageView")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .field("info", &self.inner.info)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for ImageView {}
impl PartialEq for ImageView {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for ImageView {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::ImageView,
    info: ImageViewInfo,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_image_view(self.handle) }
        }
    }
}
