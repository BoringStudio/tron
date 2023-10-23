use std::ptr::NonNull;

use gpu_alloc::{
    AllocationFlags, DeviceMapError, DeviceProperties, MappedMemoryRange, MemoryDevice, MemoryHeap,
    MemoryType, OutOfMemory,
};
use smallvec::SmallVec;
use vulkanalia::prelude::v1_0::*;

#[derive(Debug)]
#[repr(transparent)]
pub struct VkMemoryDevice {
    device: Device,
}

impl VkMemoryDevice {
    pub fn wrap(device: &Device) -> &Self {
        // SAFETY: `VkMemoryDerive` has `#[repr(transparent)]`
        unsafe { &*(device as *const Device as *const Self) }
    }
}

impl MemoryDevice<vk::DeviceMemory> for VkMemoryDevice {
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: AllocationFlags,
    ) -> Result<vk::DeviceMemory, OutOfMemory> {
        let mut info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type);

        let mut info_flags;
        if flags.contains(AllocationFlags::DEVICE_ADDRESS) {
            info_flags = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            info = info.push_next(&mut info_flags);
        }

        match self.device.allocate_memory(&info, None) {
            Ok(memory) => Ok(memory),
            Err(vk::ErrorCode::OUT_OF_DEVICE_MEMORY) => Err(OutOfMemory::OutOfDeviceMemory),
            Err(vk::ErrorCode::OUT_OF_HOST_MEMORY) => Err(OutOfMemory::OutOfHostMemory),
            Err(vk::ErrorCode::TOO_MANY_OBJECTS) => panic!("Too many objects"),
            Err(e) => panic!("Unexpected Vulkan error: {e}"),
        }
    }

    unsafe fn deallocate_memory(&self, memory: vk::DeviceMemory) {
        self.device.free_memory(memory, None)
    }

    unsafe fn map_memory(
        &self,
        memory: &mut vk::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, DeviceMapError> {
        const FLAGS: vk::MemoryMapFlags = vk::MemoryMapFlags::empty();

        match self.device.map_memory(*memory, offset, size, FLAGS) {
            Ok(ptr) => {
                Ok(NonNull::new(ptr as *mut u8)
                    .expect("Pointer to memory mapping must not be null"))
            }
            Err(vk::ErrorCode::OUT_OF_DEVICE_MEMORY) => Err(DeviceMapError::OutOfDeviceMemory),
            Err(vk::ErrorCode::OUT_OF_HOST_MEMORY) => Err(DeviceMapError::OutOfHostMemory),
            Err(vk::ErrorCode::MEMORY_MAP_FAILED) => Err(DeviceMapError::MapFailed),
            Err(e) => panic!("Unexpected Vulkan error: {e}"),
        }
    }

    unsafe fn unmap_memory(&self, memory: &mut vk::DeviceMemory) {
        self.device.unmap_memory(*memory)
    }

    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        let ranges = ranges
            .iter()
            .map(|range| {
                vk::MappedMemoryRange::builder()
                    .memory(*range.memory)
                    .offset(range.offset)
                    .size(range.size)
            })
            .collect::<SmallVec<[_; 4]>>();

        self.device
            .invalidate_mapped_memory_ranges(&ranges)
            .map_err(|e| match e {
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk::ErrorCode::OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                e => panic!("Unexpected Vulkan error: {e}"),
            })
    }

    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        let ranges = ranges
            .iter()
            .map(|range| {
                vk::MappedMemoryRange::builder()
                    .memory(*range.memory)
                    .offset(range.offset)
                    .size(range.size)
            })
            .collect::<SmallVec<[_; 4]>>();

        self.device
            .flush_mapped_memory_ranges(&ranges)
            .map_err(|e| match e {
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk::ErrorCode::OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                e => panic!("Unexpected Vulkan error: {e}"),
            })
    }
}

pub unsafe fn get_device_properties(
    instance: &Instance,
    device: vk::PhysicalDevice,
) -> anyhow::Result<DeviceProperties> {
    let limits = instance.get_physical_device_properties(device).limits;
    let memory_properties = instance.get_physical_device_memory_properties(device);

    Ok(DeviceProperties {
        max_memory_allocation_count: limits.max_memory_allocation_count,
        max_memory_allocation_size: u64::MAX,
        non_coherent_atom_size: limits.non_coherent_atom_size,
        memory_types: memory_properties.memory_types
            [..memory_properties.memory_type_count as usize]
            .iter()
            .map(|memory_type| MemoryType {
                props: map_memory_properties(memory_type.property_flags),
                heap: memory_type.heap_index,
            })
            .collect(),
        memory_heaps: memory_properties.memory_heaps
            [..memory_properties.memory_heap_count as usize]
            .iter()
            .map(|memory_heap| MemoryHeap {
                size: memory_heap.size,
            })
            .collect(),
        // TODO: check if vulkan 1.1 and set accordingly
        buffer_device_address: false,
    })
}

fn map_memory_properties(props: vk::MemoryPropertyFlags) -> gpu_alloc::MemoryPropertyFlags {
    let mut result = gpu_alloc::MemoryPropertyFlags::empty();
    if props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= gpu_alloc::MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        result |= gpu_alloc::MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        result |= gpu_alloc::MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
        result |= gpu_alloc::MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= gpu_alloc::MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}
