use std::mem::ManuallyDrop;
use std::num::NonZeroU64;
use std::sync::Arc;

use gpu_alloc::MemoryBlock;
use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;

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

impl From<vk::Extent3D> for ImageExtent {
    fn from(value: vk::Extent3D) -> Self {
        Self::D3 {
            width: value.width,
            height: value.height,
            depth: value.depth,
        }
    }
}

impl From<ImageExtent> for vk::Extent3D {
    fn from(value: ImageExtent) -> Self {
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

impl From<Samples> for vk::SampleCountFlags {
    fn from(value: Samples) -> Self {
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
pub struct ImageInfo {
    pub extent: ImageExtent,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub samples: Samples,
    pub layers: u32,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Clone)]
pub struct Image {
    info: ImageInfo,
    inner: Arc<Inner>,
}

impl Image {
    pub fn new_surface(
        handle: vk::Image,
        info: ImageInfo,
        owner: WeakDevice,
        id: NonZeroU64,
    ) -> Self {
        Self {
            info,
            inner: Arc::new(Inner {
                handle,
                owner,
                source: ImageSource::Surface { id },
            }),
        }
    }

    pub fn info(&self) -> &ImageInfo {
        &self.info
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
                .field("info", &self.info)
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
    owner: WeakDevice,
    source: ImageSource,
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
