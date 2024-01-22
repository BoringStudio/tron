use std::sync::{Arc, Mutex};

use vulkanalia::prelude::v1_0::*;

use crate::device::{AllocatedDescriptorSet, WeakDevice};
use crate::resources::{
    BufferRange, BufferView, DescriptorSetLayout, DescriptorType, ImageLayout, ImageView, Sampler,
};

/// Structure specifying how to update the contents of a descriptor set object.
pub struct UpdateDescriptorSet<'a> {
    pub set: &'a DescriptorSet,
    pub writes: &'a [DescriptorSetWrite<'a>],
}

/// Structure specifying the parameters of a descriptor set write operation.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DescriptorSetWrite<'a> {
    pub binding: u32,
    pub element: u32,
    pub data: DescriptorSlice<'a>,
}

/// A slice of descriptor data.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DescriptorSlice<'a> {
    Sampler(&'a [Sampler]),
    CombinedImageSampler(&'a [CombinedImageSampler]),
    SampledImage(&'a [(ImageView, ImageLayout)]),
    StorageImage(&'a [(ImageView, ImageLayout)]),
    UniformTexelBuffer(&'a [BufferView]),
    StorageTexelBuffer(&'a [BufferView]),
    UniformBuffer(&'a [BufferRange]),
    StorageBuffer(&'a [BufferRange]),
    UniformBufferDynamic(&'a [BufferRange]),
    StorageBufferDynamic(&'a [BufferRange]),
    InputAttachment(&'a [(ImageView, ImageLayout)]),
}

/// A descriptor which makes it possible for shaders to access an image
/// resource through a sampler object.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CombinedImageSampler {
    pub view: ImageView,
    pub layout: ImageLayout,
    pub sampler: Sampler,
}

/// Structure specifying the allocation parameters for descriptor sets.
#[derive(Debug, Clone)]
pub struct DescriptorSetInfo {
    pub layout: DescriptorSetLayout,
}

/// A wrapper around a Vulkan descriptor set object.
///
/// A pack of handles to resources that can be bound to a pipeline.
#[derive(Clone)]
pub struct DescriptorSet {
    inner: Arc<Inner>,
}

impl DescriptorSet {
    pub(crate) fn new(
        allocated: AllocatedDescriptorSet,
        info: DescriptorSetInfo,
        owner: WeakDevice,
    ) -> Self {
        let bindings = info
            .layout
            .info()
            .bindings
            .iter()
            .map(|binding| match binding.ty {
                DescriptorType::Sampler => ReferencedDescriptor::Sampler(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::CombinedImageSampler => ReferencedDescriptor::CombinedImageSampler(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::SampledImage => ReferencedDescriptor::SampledImage(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::StorageImage => ReferencedDescriptor::StorageImage(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::UniformTexelBuffer => ReferencedDescriptor::UniformTexelBuffer(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::StorageTexelBuffer => ReferencedDescriptor::StorageTexelBuffer(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::UniformBuffer => ReferencedDescriptor::UniformBuffer(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::StorageBuffer => ReferencedDescriptor::StorageBuffer(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::UniformBufferDynamic => ReferencedDescriptor::UniformBufferDynamic(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::StorageBufferDynamic => ReferencedDescriptor::StorageBufferDynamic(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
                DescriptorType::InputAttachment => ReferencedDescriptor::InputAttachment(
                    vec![None; binding.count as usize].into_boxed_slice(),
                ),
            })
            .collect();

        Self {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(Inner {
                allocated,
                info,
                owner,
                bindings: Mutex::new(bindings),
            }),
        }
    }

    pub fn handle(&self) -> vk::DescriptorSet {
        self.inner.allocated.handle()
    }

    pub fn info(&self) -> &DescriptorSetInfo {
        &self.inner.info
    }

    pub fn write_descriptors(&self, binding: u32, element: u32, data: DescriptorSlice) {
        let mut bindings = self.inner.bindings.lock().unwrap();

        match (data, &mut bindings[binding as usize]) {
            // Sampler
            (DescriptorSlice::Sampler(data), ReferencedDescriptor::Sampler(refs)) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // CombinedImageSampler
            (
                DescriptorSlice::CombinedImageSampler(data),
                ReferencedDescriptor::CombinedImageSampler(refs),
            ) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // SampledImage
            (DescriptorSlice::SampledImage(data), ReferencedDescriptor::SampledImage(refs)) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // StorageImage
            (DescriptorSlice::StorageImage(data), ReferencedDescriptor::StorageImage(refs)) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // UniformTexelBuffer
            (
                DescriptorSlice::UniformTexelBuffer(data),
                ReferencedDescriptor::UniformTexelBuffer(refs),
            ) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // StorageTexelBuffer
            (
                DescriptorSlice::StorageTexelBuffer(data),
                ReferencedDescriptor::StorageTexelBuffer(refs),
            ) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // UniformBuffer
            (DescriptorSlice::UniformBuffer(data), ReferencedDescriptor::UniformBuffer(refs)) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // StorageBuffer
            (DescriptorSlice::StorageBuffer(data), ReferencedDescriptor::StorageBuffer(refs)) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // UniformBufferDynamic
            (
                DescriptorSlice::UniformBufferDynamic(data),
                ReferencedDescriptor::UniformBufferDynamic(refs),
            ) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // StorageBufferDynamic
            (
                DescriptorSlice::StorageBufferDynamic(data),
                ReferencedDescriptor::StorageBufferDynamic(refs),
            ) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // InputAttachment
            (
                DescriptorSlice::InputAttachment(data),
                ReferencedDescriptor::InputAttachment(refs),
            ) => {
                for (slot, data) in refs.iter_mut().skip(element as usize).zip(data) {
                    *slot = Some(data.clone());
                }
            }
            // Invalid
            _ => {
                debug_assert!(false, "invalid descriptor slice");
            }
        }
    }
}

impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("DescriptorSet")
                .field("handle", &self.inner.allocated.handle())
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.allocated.handle(), f)
        }
    }
}

impl Eq for DescriptorSet {}
impl PartialEq for DescriptorSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for DescriptorSet {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    allocated: AllocatedDescriptorSet,
    info: DescriptorSetInfo,
    owner: WeakDevice,
    bindings: Mutex<Vec<ReferencedDescriptor>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_descriptor_set(&self.allocated) }
        }
    }
}

enum ReferencedDescriptor {
    Sampler(Box<[Option<Sampler>]>),
    CombinedImageSampler(Box<[Option<CombinedImageSampler>]>),
    SampledImage(Box<[Option<(ImageView, ImageLayout)>]>),
    StorageImage(Box<[Option<(ImageView, ImageLayout)>]>),
    UniformTexelBuffer(Box<[Option<BufferView>]>),
    StorageTexelBuffer(Box<[Option<BufferView>]>),
    UniformBuffer(Box<[Option<BufferRange>]>),
    StorageBuffer(Box<[Option<BufferRange>]>),
    UniformBufferDynamic(Box<[Option<BufferRange>]>),
    StorageBufferDynamic(Box<[Option<BufferRange>]>),
    InputAttachment(Box<[Option<(ImageView, ImageLayout)>]>),
}
