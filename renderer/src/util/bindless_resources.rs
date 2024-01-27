use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use anyhow::Result;

pub struct BindlessResources {
    descriptor_set_layout: gfx::DescriptorSetLayout,
    descriptor_set: gfx::DescriptorSet,

    image_allocator: ImageHandleAllocator,
    uniform_buffer_allocator: UniformBufferHandleAllocator,
    storage_buffer_allocator: StorageBufferHandleAllocator,
}

impl BindlessResources {
    #[tracing::instrument(level = "debug", name = "create_bindless_resources", skip_all)]
    pub fn new(device: &gfx::Device) -> Result<Self> {
        // Create descriptor set layout
        let flags = gfx::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | gfx::DescriptorBindingFlags::PARTIALLY_BOUND;
        let descriptor_set_layout =
            device.create_descriptor_set_layout(gfx::DescriptorSetLayoutInfo {
                bindings: vec![
                    gfx::DescriptorSetLayoutBinding {
                        binding: IMAGE_BINDING,
                        ty: gfx::DescriptorType::CombinedImageSampler,
                        count: IMAGE_CAPACITY,
                        stages: gfx::ShaderStageFlags::ALL,
                        flags,
                    },
                    gfx::DescriptorSetLayoutBinding {
                        binding: UNIFORM_BUFFER_BINDING,
                        ty: gfx::DescriptorType::UniformBuffer,
                        count: UNIFORM_BUFFER_CAPACITY,
                        stages: gfx::ShaderStageFlags::ALL,
                        flags,
                    },
                    gfx::DescriptorSetLayoutBinding {
                        binding: STORAGE_BUFFER_BINDING,
                        ty: gfx::DescriptorType::StorageBuffer,
                        count: STORAGE_BUFFER_CAPACITY,
                        stages: gfx::ShaderStageFlags::ALL,
                        flags,
                    },
                ],
                flags: gfx::DescriptorSetLayoutFlags::UPDATE_AFTER_BIND_POOL,
            })?;

        // Create descriptor set
        let descriptor_set = device.create_descriptor_set(gfx::DescriptorSetInfo {
            layout: descriptor_set_layout.clone(),
        })?;

        Ok(Self {
            descriptor_set_layout,
            descriptor_set,
            image_allocator: Default::default(),
            uniform_buffer_allocator: Default::default(),
            storage_buffer_allocator: Default::default(),
        })
    }

    pub fn descriptor_set_layout(&self) -> &gfx::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn descriptor_set(&self) -> &gfx::DescriptorSet {
        &self.descriptor_set
    }

    pub fn flush_retired(&self) {
        self.image_allocator.flush_retired();
        self.uniform_buffer_allocator.flush_retired();
        self.storage_buffer_allocator.flush_retired();
    }

    #[allow(dead_code)]
    pub fn alloc_image(
        &self,
        device: &gfx::Device,
        image: gfx::ImageView,
        sampler: gfx::Sampler,
    ) -> SampledImageHandle {
        let handle = self.image_allocator.alloc();

        device.update_descriptor_sets(&[gfx::UpdateDescriptorSet {
            set: &self.descriptor_set,
            writes: &[gfx::DescriptorSetWrite {
                binding: IMAGE_BINDING,
                element: handle.index(),
                data: gfx::DescriptorSlice::CombinedImageSampler(&[gfx::CombinedImageSampler {
                    view: image,
                    layout: gfx::ImageLayout::ShaderReadOnlyOptimal,
                    sampler,
                }]),
            }],
        }]);

        handle
    }

    #[allow(dead_code)]
    pub fn free_image(&self, handle: SampledImageHandle) {
        self.image_allocator.dealloc(handle);
    }

    #[allow(dead_code)]
    pub fn alloc_uniform_buffer(
        &self,
        device: &gfx::Device,
        buffer: gfx::Buffer,
    ) -> UniformBufferHandle {
        let handle = self.uniform_buffer_allocator.alloc();

        device.update_descriptor_sets(&[gfx::UpdateDescriptorSet {
            set: &self.descriptor_set,
            writes: &[gfx::DescriptorSetWrite {
                binding: UNIFORM_BUFFER_BINDING,
                element: handle.index(),
                data: gfx::DescriptorSlice::UniformBuffer(&[gfx::BufferRange::whole(buffer)]),
            }],
        }]);

        handle
    }

    #[allow(dead_code)]
    pub fn free_uniform_buffer(&self, handle: UniformBufferHandle) {
        self.uniform_buffer_allocator.dealloc(handle);
    }

    pub fn alloc_storage_buffer(
        &self,
        device: &gfx::Device,
        buffer: gfx::Buffer,
    ) -> StorageBufferHandle {
        let handle = self.storage_buffer_allocator.alloc();

        device.update_descriptor_sets(&[gfx::UpdateDescriptorSet {
            set: &self.descriptor_set,
            writes: &[gfx::DescriptorSetWrite {
                binding: STORAGE_BUFFER_BINDING,
                element: handle.index(),
                data: gfx::DescriptorSlice::StorageBuffer(&[gfx::BufferRange::whole(buffer)]),
            }],
        }]);

        handle
    }

    pub fn free_storage_buffer(&self, handle: StorageBufferHandle) {
        self.storage_buffer_allocator.dealloc(handle);
    }
}

#[repr(u8)]
pub enum GpuResourceKind {
    UniformBuffer = 0,
    StorageBuffer = 1,
    SampledImage = 2,
}

type UniformBufferHandleAllocator =
    GpuResourceHandleAllocator<{ GpuResourceKind::UniformBuffer as u8 }>;
type StorageBufferHandleAllocator =
    GpuResourceHandleAllocator<{ GpuResourceKind::StorageBuffer as u8 }>;
type ImageHandleAllocator = GpuResourceHandleAllocator<{ GpuResourceKind::SampledImage as u8 }>;

/// Allocator for GPU resource handles with two-stage deallocation.
///
/// When a handle is deallocated, it is not immediately returned to the free list,
/// but instead is added to the retired list. The free list is only updated when
/// `flush_retired` is called.
#[derive(Default)]
struct GpuResourceHandleAllocator<const KIND: u8> {
    next_index: AtomicU32,
    unused_handles: Mutex<UnusedHandles>,
}

impl<const KIND: u8> GpuResourceHandleAllocator<KIND> {
    fn alloc(&self) -> GpuResourceHandle<KIND> {
        fn alloc_impl(
            kind: u8,
            next_index: &AtomicU32,
            unused_handles: &Mutex<UnusedHandles>,
        ) -> u32 {
            match unused_handles.lock().unwrap().free_list.pop() {
                Some(handle) => recycle_handle(handle),
                None => {
                    let index = next_index.fetch_add(1, Ordering::Relaxed);
                    ((kind as u32 & HANDLE_KIND_MASK) << HANDLE_KIND_OFFSET)
                        | (index & HANDLE_INDEX_MASK)
                }
            }
        }
        GpuResourceHandle(alloc_impl(KIND, &self.next_index, &self.unused_handles))
    }

    fn dealloc(&self, handle: GpuResourceHandle<KIND>) {
        self.unused_handles
            .lock()
            .unwrap()
            .retired_list
            .push(handle.0);
    }

    fn flush_retired(&self) {
        fn flush_retired_impl(unused_handles: &Mutex<UnusedHandles>) {
            let mut handles = unused_handles.lock().unwrap();
            let handles = &mut *handles;
            handles.free_list.append(&mut handles.retired_list);
            // TODO: sort free list?
        }
        flush_retired_impl(&self.unused_handles)
    }
}

#[derive(Default)]
struct UnusedHandles {
    free_list: Vec<u32>,
    retired_list: Vec<u32>,
}

pub type UniformBufferHandle = GpuResourceHandle<{ GpuResourceKind::UniformBuffer as u8 }>;
pub type StorageBufferHandle = GpuResourceHandle<{ GpuResourceKind::StorageBuffer as u8 }>;
pub type SampledImageHandle = GpuResourceHandle<{ GpuResourceKind::SampledImage as u8 }>;

#[derive(Clone, Copy, Hash, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct GpuResourceHandle<const KIND: u8>(u32);

// TODO: add resource type validation
impl<const KIND: u8> GpuResourceHandle<KIND> {
    pub const INVALID: Self = Self::new(0, u32::MAX);

    pub const fn new(version: u8, index: u32) -> Self {
        Self(
            (version as u32) << HANDLE_VERSION_OFFSET
                | (((KIND as u32) & HANDLE_KIND_MASK) << HANDLE_KIND_OFFSET)
                | (index & HANDLE_INDEX_MASK),
        )
    }

    pub fn version(self) -> u8 {
        (self.0 >> HANDLE_VERSION_OFFSET) as u8
    }

    pub fn index(self) -> u32 {
        self.0 & HANDLE_INDEX_MASK
    }
}

impl<const KIND: u8> std::fmt::Debug for GpuResourceHandle<KIND> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuResourceHandle")
            .field("version", &self.version())
            .field("kind", &KIND)
            .field("index", &self.index())
            .finish()
    }
}

const fn recycle_handle(handle: u32) -> u32 {
    handle.wrapping_add(1 << HANDLE_VERSION_OFFSET)
}

const HANDLE_KIND_BITS: usize = 2;
const HANDLE_INDEX_BITS: usize = 24;

const HANDLE_VERSION_OFFSET: usize = HANDLE_KIND_BITS + HANDLE_INDEX_BITS;
const HANDLE_KIND_OFFSET: usize = HANDLE_INDEX_BITS;
const HANDLE_KIND_MASK: u32 = (1 << HANDLE_KIND_BITS) - 1;
const HANDLE_INDEX_MASK: u32 = (1 << HANDLE_INDEX_BITS) - 1;

const IMAGE_BINDING: u32 = 0;
const UNIFORM_BUFFER_BINDING: u32 = 1;
const STORAGE_BUFFER_BINDING: u32 = 2;

const IMAGE_CAPACITY: u32 = 1024;
const UNIFORM_BUFFER_CAPACITY: u32 = 1024;
const STORAGE_BUFFER_CAPACITY: u32 = 1024;
