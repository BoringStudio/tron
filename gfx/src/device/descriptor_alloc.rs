use std::collections::VecDeque;

use shared::util::WithDefer;
use shared::FastHashMap;
use smallvec::SmallVec;
use vulkanalia::prelude::v1_0::*;

use crate::resources::{DescriptorSetLayout, DescriptorSetLayoutFlags, DescriptorSetSize};
use crate::OutOfDeviceMemory;

#[derive(Default)]
pub(crate) struct DescriptorAlloc {
    buckets: FastHashMap<(DescriptorSetSize, bool), DescriptorBucket>,
    sets_cache: Vec<AllocatedDescriptorSet>,
    raw_sets_cache: Vec<vk::DescriptorSet>,
}

impl DescriptorAlloc {
    pub fn new() -> Self {
        Self::default()
    }

    pub unsafe fn allocate(
        &mut self,
        device: &Device,
        layout: &DescriptorSetLayout,
        count: u32,
    ) -> Result<Vec<AllocatedDescriptorSet>, DescriptorAllocError> {
        if count == 0 {
            return Ok(Default::default());
        }

        let info = layout.info();

        let update_after_bind = info
            .flags
            .contains(DescriptorSetLayoutFlags::UPDATE_AFTER_BIND_POOL);

        let bucket = self
            .buckets
            .entry((*layout.size(), update_after_bind))
            .or_insert_with(|| DescriptorBucket::new(update_after_bind, layout.size()));

        match bucket.allocate(device, layout, count, &mut self.sets_cache) {
            Ok(()) => Ok(std::mem::take(&mut self.sets_cache)),
            Err(e) => {
                if let Some(mut last_pool_id) = self.sets_cache.first().map(|s| s.pool_id) {
                    for set in &self.sets_cache {
                        if set.pool_id != last_pool_id {
                            bucket.free(device, &self.raw_sets_cache, last_pool_id);

                            self.raw_sets_cache.clear();
                            last_pool_id = set.pool_id;
                        }
                        self.raw_sets_cache.push(set.handle);
                    }

                    if !self.raw_sets_cache.is_empty() {
                        bucket.free(device, &self.raw_sets_cache, last_pool_id);
                        self.raw_sets_cache.clear();
                    }
                }

                self.sets_cache.clear();
                Err(e)
            }
        }
    }

    pub unsafe fn free(&mut self, device: &Device, sets: &[AllocatedDescriptorSet]) {
        let (mut last_key, mut last_pool_id) = match sets.first() {
            Some(set) => ((set.size, set.update_after_bind), set.pool_id),
            None => return,
        };

        for set in sets {
            if last_key != (set.size, set.update_after_bind) || last_pool_id != set.pool_id {
                self.buckets
                    .get_mut(&last_key)
                    .expect("invalid bucket key")
                    .free(device, &self.raw_sets_cache, last_pool_id);

                self.raw_sets_cache.clear();
                last_key = (set.size, set.update_after_bind);
                last_pool_id = set.pool_id;
            }

            self.raw_sets_cache.push(set.handle);
        }

        if !self.raw_sets_cache.is_empty() {
            self.buckets
                .get_mut(&last_key)
                .expect("invalid bucket key")
                .free(device, &self.raw_sets_cache, last_pool_id);

            self.raw_sets_cache.clear();
        }
    }

    pub unsafe fn cleanup(&mut self, device: &Device) {
        for bucket in self.buckets.values_mut() {
            bucket.cleanup(device);
        }
        self.buckets.retain(|_, bucket| !bucket.pools.is_empty());
    }
}

impl Drop for DescriptorAlloc {
    fn drop(&mut self) {
        if self.buckets.drain().any(|(_, bucket)| bucket.total > 0) {
            tracing::error!("allocator is dropped while some descriptor sets are still allocated");
        }
    }
}

pub(crate) struct AllocatedDescriptorSet {
    handle: vk::DescriptorSet,
    size: DescriptorSetSize,
    pool_id: u64,
    update_after_bind: bool,
}

impl AllocatedDescriptorSet {
    pub fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }
}

struct DescriptorBucket {
    pools: VecDeque<DescriptorPool>,
    offset: u64,
    total: u64,
    update_after_bind: bool,
    size: DescriptorSetSize,
}

impl DescriptorBucket {
    fn new(update_after_bind: bool, size: &DescriptorSetSize) -> Self {
        Self {
            pools: VecDeque::new(),
            offset: 0,
            total: 0,
            update_after_bind,
            size: *size,
        }
    }

    unsafe fn allocate(
        &mut self,
        device: &Device,
        layout: &DescriptorSetLayout,
        mut count: u32,
        allocated_sets: &mut Vec<AllocatedDescriptorSet>,
    ) -> Result<(), DescriptorAllocError> {
        fn extend_allocated_sets(
            update_after_bind: bool,
            size: &DescriptorSetSize,
            pool_id: u64,
            handles: &[vk::DescriptorSet],
            allocated_sets: &mut Vec<AllocatedDescriptorSet>,
        ) {
            allocated_sets.extend(handles.iter().map(|&handle| AllocatedDescriptorSet {
                handle,
                size: *size,
                pool_id,
                update_after_bind,
            }))
        }

        if count == 0 {
            return Ok(());
        }

        let mut set_layouts = SmallVec::<[_; 16]>::new();

        // Allocate from existing pools
        for (i, pool) in self.pools.iter_mut().enumerate() {
            if pool.remaining == 0 {
                continue;
            }

            let allocate = pool.remaining.min(count);
            tracing::trace!(
                descriptor_pool = ?pool.handle,
                count = allocate,
                "allocating descriptor sets from an existing pool",
            );

            set_layouts.resize_with(allocate as usize, || layout.handle());

            let new_sets = match device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(pool.handle)
                    .set_layouts(&set_layouts),
            ) {
                Ok(new_sets) => new_sets,
                Err(vk::ErrorCode::OUT_OF_DEVICE_MEMORY) => {
                    return Err(DescriptorAllocError::OutOfDeviceMemory(OutOfDeviceMemory))
                }
                Err(vk::ErrorCode::OUT_OF_HOST_MEMORY) => crate::out_of_host_memory(),
                Err(vk::ErrorCode::FRAGMENTED_POOL) => {
                    tracing::error!(
                        descriptor_pool = ?pool.handle,
                        "failed to allocate descriptor sets due to pool fragmentation",
                    );
                    pool.remaining = 0;
                    continue;
                }
                Err(vk::ErrorCode::OUT_OF_POOL_MEMORY) => {
                    pool.remaining = 0;
                    continue;
                }
                Err(e) => crate::unexpected_vulkan_error(e),
            };

            extend_allocated_sets(
                self.update_after_bind,
                &self.size,
                self.offset + i as u64,
                &new_sets,
                allocated_sets,
            );
            count -= allocate;
            pool.allocated += allocate;
            pool.remaining -= allocate;
            self.total += allocate as u64;

            if count == 0 {
                return Ok(());
            }
        }

        while count > 0 {
            let (pool_size, max_sets) = self.next_pool_size(count);
            tracing::trace!(?pool_size, max_sets, "creating a new descriptor pool");

            let handle = create_descriptor_pool(
                device,
                &pool_size,
                max_sets,
                if self.update_after_bind {
                    vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                        | vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
                } else {
                    vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                },
            )?
            .with_defer(|pool| device.destroy_descriptor_pool(pool, None));

            let allocate = max_sets.min(count);
            tracing::trace!(
                descriptor_pool = ?*handle,
                count = allocate,
                "allocating descriptor sets from a new pool",
            );

            set_layouts.resize_with(allocate as usize, || layout.handle());

            let new_sets = match device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*handle)
                    .set_layouts(&set_layouts),
            ) {
                Ok(new_sets) => new_sets,
                Err(vk::ErrorCode::OUT_OF_DEVICE_MEMORY) => {
                    return Err(DescriptorAllocError::OutOfDeviceMemory(OutOfDeviceMemory))
                }
                Err(vk::ErrorCode::OUT_OF_HOST_MEMORY) => crate::out_of_host_memory(),
                Err(e) => crate::unexpected_vulkan_error(e),
            };

            extend_allocated_sets(
                self.update_after_bind,
                &self.size,
                self.offset + self.pools.len() as u64,
                &new_sets,
                allocated_sets,
            );

            count -= allocate;
            self.pools.push_back(DescriptorPool {
                handle: handle.disarm(),
                allocated: allocate,
                remaining: max_sets - allocate,
            });
            self.total += allocate as u64;
        }

        Ok(())
    }

    unsafe fn free(
        &mut self,
        device: &Device,
        descriptor_sets: &[vk::DescriptorSet],
        pool_id: u64,
    ) {
        let pool = pool_id
            .checked_sub(self.offset)
            .and_then(|i| self.pools.get_mut(i as usize))
            .expect("invalid descriptor pool id");

        tracing::trace!(descriptor_pool = ?pool.handle, ?descriptor_sets, "deallocating descriptor sets");
        device
            .free_descriptor_sets(pool.handle, descriptor_sets)
            .unwrap();

        let deallocated = descriptor_sets.len() as u32;
        pool.allocated -= deallocated;
        pool.remaining += deallocated;
        self.total -= deallocated as u64;

        while self.pools.len() > 1 {
            let pool = match self.pools.front_mut() {
                Some(pool) if pool.allocated == 0 => pool,
                _ => break,
            };

            tracing::trace!(descriptor_pool = ?pool.handle, "destroying an empty descriptor pool");
            device.destroy_descriptor_pool(pool.handle, None);

            self.offset += 1;
            self.pools.pop_front();
        }
    }

    unsafe fn cleanup(&mut self, device: &Device) {
        loop {
            let pool = match self.pools.front_mut() {
                Some(pool) if pool.allocated == 0 => pool,
                _ => break,
            };

            tracing::trace!(descriptor_pool = ?pool.handle, "destroying an empty descriptor pool");
            device.destroy_descriptor_pool(pool.handle, None);

            self.offset += 1;
            self.pools.pop_front();
        }
    }

    fn next_pool_size(&self, required: u32) -> (DescriptorSetSize, u32) {
        let mut max_sets = MIN_SETS
            .max(required)
            .max(self.total.min(MAX_SETS as u64) as u32)
            .checked_next_power_of_two()
            .unwrap_or(i32::MAX as u32);

        // Prevent any part from decreasing to less than its current value
        max_sets = (u32::MAX / self.size.samplers.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.combined_image_samplers.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.sampled_images.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_images.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.uniform_texel_buffers.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_texel_buffers.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.uniform_buffers.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_buffers.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.uniform_buffers_dynamic.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.storage_buffers_dynamic.max(1)).min(max_sets);
        max_sets = (u32::MAX / self.size.input_attachments.max(1)).min(max_sets);

        let mut res = DescriptorSetSize {
            samplers: self.size.samplers * max_sets,
            combined_image_samplers: self.size.combined_image_samplers * max_sets,
            sampled_images: self.size.sampled_images * max_sets,
            storage_images: self.size.storage_images * max_sets,
            uniform_texel_buffers: self.size.uniform_texel_buffers * max_sets,
            storage_texel_buffers: self.size.storage_texel_buffers * max_sets,
            uniform_buffers: self.size.uniform_buffers * max_sets,
            storage_buffers: self.size.storage_buffers * max_sets,
            uniform_buffers_dynamic: self.size.uniform_buffers_dynamic * max_sets,
            storage_buffers_dynamic: self.size.storage_buffers_dynamic * max_sets,
            input_attachments: self.size.input_attachments * max_sets,
        };

        if res == DescriptorSetSize::ZERO {
            res.samplers += 1;
        }

        (res, max_sets)
    }
}

impl Drop for DescriptorBucket {
    fn drop(&mut self) {
        if self.total > 0 {
            tracing::error!("descriptor sets leaked");
        }
    }
}

#[derive(Debug)]
struct DescriptorPool {
    handle: vk::DescriptorPool,
    allocated: u32,
    remaining: u32,
}

unsafe fn create_descriptor_pool(
    device: &Device,
    size: &DescriptorSetSize,
    max_sets: u32,
    flags: vk::DescriptorPoolCreateFlags,
) -> Result<vk::DescriptorPool, DescriptorAllocError> {
    let mut array = [vk::DescriptorPoolSize::builder().build(); 11];
    let mut len = 0;

    if size.samplers != 0 {
        array[len].type_ = vk::DescriptorType::SAMPLER;
        array[len].descriptor_count = size.samplers;
        len += 1;
    }

    if size.combined_image_samplers != 0 {
        array[len].type_ = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        array[len].descriptor_count = size.combined_image_samplers;
        len += 1;
    }

    if size.sampled_images != 0 {
        array[len].type_ = vk::DescriptorType::SAMPLED_IMAGE;
        array[len].descriptor_count = size.sampled_images;
        len += 1;
    }

    if size.storage_images != 0 {
        array[len].type_ = vk::DescriptorType::STORAGE_IMAGE;
        array[len].descriptor_count = size.storage_images;
        len += 1;
    }

    if size.uniform_texel_buffers != 0 {
        array[len].type_ = vk::DescriptorType::UNIFORM_TEXEL_BUFFER;
        array[len].descriptor_count = size.uniform_texel_buffers;
        len += 1;
    }

    if size.storage_texel_buffers != 0 {
        array[len].type_ = vk::DescriptorType::STORAGE_TEXEL_BUFFER;
        array[len].descriptor_count = size.storage_texel_buffers;
        len += 1;
    }

    if size.uniform_buffers != 0 {
        array[len].type_ = vk::DescriptorType::UNIFORM_BUFFER;
        array[len].descriptor_count = size.uniform_buffers;
        len += 1;
    }

    if size.storage_buffers != 0 {
        array[len].type_ = vk::DescriptorType::STORAGE_BUFFER;
        array[len].descriptor_count = size.storage_buffers;
        len += 1;
    }

    if size.uniform_buffers_dynamic != 0 {
        array[len].type_ = vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC;
        array[len].descriptor_count = size.uniform_buffers_dynamic;
        len += 1;
    }

    if size.storage_buffers_dynamic != 0 {
        array[len].type_ = vk::DescriptorType::STORAGE_BUFFER_DYNAMIC;
        array[len].descriptor_count = size.storage_buffers_dynamic;
        len += 1;
    }

    if size.input_attachments != 0 {
        array[len].type_ = vk::DescriptorType::INPUT_ATTACHMENT;
        array[len].descriptor_count = size.input_attachments;
        len += 1;
    }

    match device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(&array[..len])
            .flags(flags),
        None,
    ) {
        Ok(handle) => Ok(handle),
        Err(vk::ErrorCode::OUT_OF_HOST_MEMORY) => crate::out_of_host_memory(),
        Err(vk::ErrorCode::OUT_OF_DEVICE_MEMORY) => {
            Err(DescriptorAllocError::OutOfDeviceMemory(OutOfDeviceMemory))
        }
        Err(vk::ErrorCode::FRAGMENTATION) => Err(DescriptorAllocError::Fragmentation),
        Err(e) => crate::unexpected_vulkan_error(e),
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DescriptorAllocError {
    #[error(transparent)]
    OutOfDeviceMemory(#[from] OutOfDeviceMemory),
    #[error("a pool allocation has failed due to fragmentation of the pool's memory")]
    Fragmentation,
}

const MIN_SETS: u32 = 64;
const MAX_SETS: u32 = 512;
