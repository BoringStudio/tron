use std::sync::Arc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::device::{Device, WeakDevice};
use crate::resources::{FormatExt, Image, ImageExtent, ImageInfo};

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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageViewType {
    D1,
    D2,
    D3,
    Cube,
}

impl From<ImageViewType> for vk::ImageViewType {
    fn from(value: ImageViewType) -> Self {
        match value {
            ImageViewType::D1 => vk::ImageViewType::_1D,
            ImageViewType::D2 => vk::ImageViewType::_2D,
            ImageViewType::D3 => vk::ImageViewType::_3D,
            ImageViewType::Cube => vk::ImageViewType::CUBE,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Swizzle {
    #[default]
    Identity,
    Zero,
    One,
    R,
    G,
    B,
    A,
}

impl From<Swizzle> for vk::ComponentSwizzle {
    fn from(value: Swizzle) -> Self {
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

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ComponentMapping {
    pub r: Swizzle,
    pub g: Swizzle,
    pub b: Swizzle,
    pub a: Swizzle,
}

impl From<ComponentMapping> for vk::ComponentMapping {
    fn from(value: ComponentMapping) -> Self {
        Self {
            r: value.r.into(),
            g: value.g.into(),
            b: value.b.into(),
            a: value.a.into(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImageViewInfo {
    pub ty: ImageViewType,
    pub range: vk::ImageSubresourceRange,
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
            range: Self::make_whole_image_subresource_range(image_info),
            image,
            mapping: Default::default(),
        }
    }

    pub fn is_whole_image(&self, image: &Image) -> bool {
        self.image == *image
            && self.range == Self::make_whole_image_subresource_range(image.info())
            && self.mapping == ComponentMapping::default()
            && matches!(
                (self.ty, &image.info().extent),
                (ImageViewType::D1, ImageExtent::D1 { .. })
                    | (ImageViewType::D2, ImageExtent::D2 { .. })
                    | (ImageViewType::D3, ImageExtent::D3 { .. })
            )
    }

    fn make_whole_image_subresource_range(info: &ImageInfo) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: info.format.aspect_flags(),
            base_mip_level: 0,
            level_count: info.mip_levels,
            base_array_layer: 0,
            layer_count: info.array_layers,
        }
    }
}

#[derive(Clone)]
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
