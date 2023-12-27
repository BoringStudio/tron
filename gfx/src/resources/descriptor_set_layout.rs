use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::resources::ShaderStageFlags;
use crate::util::FromGfx;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorSetLayoutInfo {
    pub bindings: Vec<DescriptorSetLayoutBinding>,
    pub flags: DescriptorSetLayoutFlags,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub ty: DescriptorType,
    pub count: u32,
    pub stages: ShaderStageFlags,
    pub flags: DescriptorBindingFlags,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DescriptorType {
    /// Contains `Sampler`.
    Sampler,
    /// Contains `ImageView` and `Sampler`.
    CombinedImageSampler,
    /// Contains `ImageView` that sampling operations can be performed on.
    SampledImage,
    /// Contains `ImageView` that load, store, and atomic operations can be performed on.
    StorageImage,
    /// Contains `BufferView` that image sampling operations can be performed on.
    UniformTexelBuffer,
    /// Contains `BufferView` that image load, store, and atomic operations can be performed on.
    StorageTexelBuffer,
    /// Contains `Buffer` that load operations can be performed on.
    UniformBuffer,
    /// Contains `Buffer` that load, store, and atomic operations can be performed on
    StorageBuffer,
    /// Same as `UniformBuffer`, but has a dynamic offset when binding the descriptor set.
    UniformBufferDynamic,
    /// Same as `StorageBuffer`, but has a dynamic offset when binding the descriptor set.
    StorageBufferDynamic,
    /// Contains `ImageView` that can be used for framebuffer local load operations in fragment shaders.
    InputAttachment,
}

impl FromGfx<DescriptorType> for vk::DescriptorType {
    fn from_gfx(value: DescriptorType) -> Self {
        match value {
            DescriptorType::Sampler => Self::SAMPLER,
            DescriptorType::CombinedImageSampler => Self::COMBINED_IMAGE_SAMPLER,
            DescriptorType::SampledImage => Self::SAMPLED_IMAGE,
            DescriptorType::StorageImage => Self::STORAGE_IMAGE,
            DescriptorType::UniformTexelBuffer => Self::UNIFORM_TEXEL_BUFFER,
            DescriptorType::StorageTexelBuffer => Self::STORAGE_TEXEL_BUFFER,
            DescriptorType::UniformBuffer => Self::UNIFORM_BUFFER,
            DescriptorType::StorageBuffer => Self::STORAGE_BUFFER,
            DescriptorType::UniformBufferDynamic => Self::UNIFORM_BUFFER_DYNAMIC,
            DescriptorType::StorageBufferDynamic => Self::STORAGE_BUFFER_DYNAMIC,
            DescriptorType::InputAttachment => Self::INPUT_ATTACHMENT,
        }
    }
}

bitflags::bitflags! {
    #[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct DescriptorBindingFlags: u32 {
        const UPDATE_AFTER_BIND = 1;
        const UPDATE_UNUSED_WHILE_PENDING = 1 << 1;
        const PARTIALLY_BOUND = 1 << 2;
        const VARIABLE_DESCRIPTOR_COUNT = 1 << 3;
    }
}

impl FromGfx<DescriptorBindingFlags> for vk::DescriptorBindingFlags {
    fn from_gfx(value: DescriptorBindingFlags) -> Self {
        let mut res = Self::empty();
        if value.contains(DescriptorBindingFlags::UPDATE_AFTER_BIND) {
            res |= Self::UPDATE_AFTER_BIND;
        }
        if value.contains(DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING) {
            res |= Self::UPDATE_UNUSED_WHILE_PENDING;
        }
        if value.contains(DescriptorBindingFlags::PARTIALLY_BOUND) {
            res |= Self::PARTIALLY_BOUND;
        }
        if value.contains(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT) {
            res |= Self::VARIABLE_DESCRIPTOR_COUNT
        }
        res
    }
}

bitflags::bitflags! {
    #[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct DescriptorSetLayoutFlags: u32 {
        const PUSH_DESCRIPTOR = 1;
        const UPDATE_AFTER_BIND_POOL = 1 << 1;
    }
}

impl FromGfx<DescriptorSetLayoutFlags> for vk::DescriptorSetLayoutCreateFlags {
    fn from_gfx(value: DescriptorSetLayoutFlags) -> Self {
        let mut res = Self::empty();
        if value.contains(DescriptorSetLayoutFlags::PUSH_DESCRIPTOR) {
            res |= Self::PUSH_DESCRIPTOR_KHR;
        }
        if value.contains(DescriptorSetLayoutFlags::UPDATE_AFTER_BIND_POOL) {
            res |= Self::UPDATE_AFTER_BIND_POOL;
        }
        res
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DescriptorSetSize {
    pub samplers: u32,
    pub combined_image_samplers: u32,
    pub sampled_images: u32,
    pub storage_images: u32,
    pub uniform_texel_buffers: u32,
    pub storage_texel_buffers: u32,
    pub uniform_buffers: u32,
    pub storage_buffers: u32,
    pub uniform_buffers_dynamic: u32,
    pub storage_buffers_dynamic: u32,
    pub input_attachments: u32,
}

impl DescriptorSetSize {
    pub const ZERO: Self = Self {
        samplers: 0,
        combined_image_samplers: 0,
        sampled_images: 0,
        storage_images: 0,
        uniform_texel_buffers: 0,
        storage_texel_buffers: 0,
        uniform_buffers: 0,
        storage_buffers: 0,
        uniform_buffers_dynamic: 0,
        storage_buffers_dynamic: 0,
        input_attachments: 0,
    };
}

#[derive(Clone)]
pub struct DescriptorSetLayout {
    inner: Arc<Inner>,
}

impl DescriptorSetLayout {
    pub(crate) fn new(
        handle: vk::DescriptorSetLayout,
        info: DescriptorSetLayoutInfo,
        size: DescriptorSetSize,
        owner: WeakDevice,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                size,
                owner,
            }),
        }
    }

    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.inner.handle
    }

    pub fn info(&self) -> &DescriptorSetLayoutInfo {
        &self.inner.info
    }

    pub fn size(&self) -> &DescriptorSetSize {
        &self.inner.size
    }
}

impl std::fmt::Debug for DescriptorSetLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("DescriptorSetLayout")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for DescriptorSetLayout {}
impl PartialEq for DescriptorSetLayout {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for DescriptorSetLayout {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::DescriptorSetLayout,
    info: DescriptorSetLayoutInfo,
    size: DescriptorSetSize,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_descriptor_set_layout(self.handle) }
        }
    }
}
