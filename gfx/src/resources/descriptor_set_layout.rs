use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::WeakDevice;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorSetLayoutInfo {
    pub bindings: Vec<DescriptorSetLayoutBinding>,
    pub flags: vk::DescriptorSetLayoutCreateFlags,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub ty: DescriptorType,
    pub count: u32,
    pub stages: vk::ShaderStageFlags,
    pub flags: vk::DescriptorBindingFlags,
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

impl From<DescriptorType> for vk::DescriptorType {
    fn from(value: DescriptorType) -> Self {
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

#[derive(Clone)]
pub struct DescriptorSetLayout {
    inner: Arc<Inner>,
}

impl DescriptorSetLayout {
    pub(crate) fn new(
        handle: vk::DescriptorSetLayout,
        info: DescriptorSetLayoutInfo,
        owner: WeakDevice,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
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
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_descriptor_set_layout(self.handle) }
        }
    }
}
