use std::cell::UnsafeCell;
use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::{AllocatedDescriptorSet, WeakDevice};
use crate::resources::{
    BufferRange, BufferView, DescriptorSetLayout, DescriptorType, ImageLayout, ImageView, Sampler,
};

/// Structure specifying how to update the contents of a descriptor set object.
pub struct UpdateDescriptorSet<'a> {
    pub set: &'a mut WritableDescriptorSet,
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

/// A unique and writable instance of a [`DescriptorSet`].
#[repr(transparent)]
pub struct WritableDescriptorSet {
    // NOTE: the struct itself must not be clonnable.
    inner: Arc<UnsafeCell<Inner>>,
}

unsafe impl Send for WritableDescriptorSet {}
unsafe impl Sync for WritableDescriptorSet {}

impl WritableDescriptorSet {
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
            inner: Arc::new(UnsafeCell::new(Inner {
                allocated,
                info,
                owner,
                bindings,
            })),
        }
    }

    unsafe fn wrap(inner: &mut Arc<UnsafeCell<Inner>>) -> &mut Self {
        &mut *(inner as *mut Arc<UnsafeCell<Inner>>).cast::<Self>()
    }

    pub fn handle(&self) -> vk::DescriptorSet {
        self.inner().allocated.handle()
    }

    pub fn info(&self) -> &DescriptorSetInfo {
        &self.inner().info
    }

    pub fn freeze(self) -> DescriptorSet {
        DescriptorSet { inner: self.inner }
    }

    pub fn write_descriptors(&mut self, binding: u32, element: u32, data: DescriptorSlice) {
        let inner = self.inner_mut();
        match (data, &mut inner.bindings[binding as usize]) {
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

    fn inner(&self) -> &Inner {
        // SAFETY: "no mutable references" is guaranteed by the interface
        unsafe { &*self.inner.get() }
    }

    fn inner_mut(&mut self) -> &mut Inner {
        // SAFETY: unique access is guaranteed by the interface
        unsafe { &mut *self.inner.get() }
    }
}

impl std::fmt::Debug for WritableDescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner();
        if f.alternate() {
            f.debug_struct("WritableDescriptorSet")
                .field("handle", &inner.allocated.handle())
                .field("owner", &inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&inner.allocated.handle(), f)
        }
    }
}

impl Eq for WritableDescriptorSet {}
impl PartialEq for WritableDescriptorSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for WritableDescriptorSet {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

/// A wrapper around a Vulkan descriptor set object.
///
/// A pack of handles to resources that can be bound to a pipeline.
#[derive(Clone)]
pub struct DescriptorSet {
    inner: Arc<UnsafeCell<Inner>>,
}

unsafe impl Send for DescriptorSet {}
unsafe impl Sync for DescriptorSet {}

impl DescriptorSet {
    pub fn handle(&self) -> vk::DescriptorSet {
        self.inner().allocated.handle()
    }

    pub fn info(&self) -> &DescriptorSetInfo {
        &self.inner().info
    }

    pub fn try_into_mut(mut self) -> Result<WritableDescriptorSet, Self> {
        if Arc::get_mut(&mut self.inner).is_some() {
            Ok(WritableDescriptorSet { inner: self.inner })
        } else {
            Err(self)
        }
    }

    pub fn get_mut(&mut self) -> Option<&mut WritableDescriptorSet> {
        if Arc::get_mut(&mut self.inner).is_some() {
            // SAFETY: descriptor set is unique
            Some(unsafe { WritableDescriptorSet::wrap(&mut self.inner) })
        } else {
            None
        }
    }

    fn inner(&self) -> &Inner {
        // SAFETY: "no mutable references" is guaranteed by the interface
        unsafe { &*self.inner.get() }
    }
}

impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner();
        if f.alternate() {
            f.debug_struct("DescriptorSet")
                .field("handle", &inner.allocated.handle())
                .field("owner", &inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&inner.allocated.handle(), f)
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
    bindings: Vec<ReferencedDescriptor>,
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
