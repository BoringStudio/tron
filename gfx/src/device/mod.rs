use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, Weak};

use bumpalo::Bump;
use gpu_alloc::{GpuAllocator, MemoryBlock};
use gpu_alloc_vulkanalia::AsMemoryDevice;
use shared::util::WithDefer;
use shared::FastDashMap;
use smallvec::SmallVec;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{DeviceV1_1, DeviceV1_2};
use winit::window::Window;

pub(crate) use self::descriptor_alloc::AllocatedDescriptorSet;
pub use self::descriptor_alloc::DescriptorAllocError;

use self::descriptor_alloc::DescriptorAlloc;
use self::epochs::Epochs;
use crate::graphics::Graphics;
use crate::physical_device::{DeviceFeatures, DeviceProperties};
use crate::queue::QueueId;
use crate::resources::{
    Blending, Buffer, BufferInfo, BufferUsage, BufferView, BufferViewInfo, ColorBlend,
    ComponentMask, ComputePipeline, ComputePipelineInfo, DescriptorBindingFlags, DescriptorSetInfo,
    DescriptorSetLayout, DescriptorSetLayoutFlags, DescriptorSetLayoutInfo, DescriptorSetSize,
    DescriptorSlice, DescriptorType, Fence, FenceState, Framebuffer, FramebufferInfo,
    GraphicsPipeline, GraphicsPipelineInfo, Image, ImageInfo, ImageView, ImageViewInfo,
    ImageViewType, MappableBuffer, MemoryUsage, PipelineLayout, PipelineLayoutInfo, RenderPass,
    RenderPassInfo, Sampler, SamplerInfo, Semaphore, ShaderModule, ShaderModuleInfo, StencilTest,
    UpdateDescriptorSet, WritableDescriptorSet,
};
use crate::surface::{CreateSurfaceError, Surface};
use crate::types::{DeviceAddress, DeviceLost, OutOfDeviceMemory, State};
use crate::util::{FromGfx, ToVk};

mod descriptor_alloc;
mod epochs;

/// A weak reference to a [`Device`].
#[derive(Clone)]
#[repr(transparent)]
pub struct WeakDevice(Weak<Inner>);

impl std::fmt::Debug for WeakDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.upgrade() {
            Some(device) => std::fmt::Debug::fmt(&device, f),
            None => write!(f, "Device({:?}, Destroyed)", self.0.as_ptr()),
        }
    }
}

impl WeakDevice {
    pub fn upgrade(&self) -> Option<Device> {
        self.0.upgrade().map(|inner| Device { inner })
    }

    pub fn is(&self, device: &Device) -> bool {
        std::ptr::eq(self.0.as_ptr(), &*device.inner)
    }
}

impl PartialEq<WeakDevice> for WeakDevice {
    fn eq(&self, other: &WeakDevice) -> bool {
        std::ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

impl PartialEq<WeakDevice> for &WeakDevice {
    fn eq(&self, other: &WeakDevice) -> bool {
        std::ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

/// A wrapper around a Vulkan logical device.
#[derive(Clone)]
#[repr(transparent)]
pub struct Device {
    inner: Arc<Inner>,
}

impl Device {
    pub(crate) fn new(
        logical: vulkanalia::Device,
        physical: vk::PhysicalDevice,
        properties: DeviceProperties,
        features: DeviceFeatures,
        queues: impl IntoIterator<Item = QueueId>,
    ) -> Self {
        let allocator = Mutex::new(GpuAllocator::new(
            gpu_alloc::Config::i_am_prototyping(),
            map_memory_device_properties(&properties, &features),
        ));
        let descriptors = Mutex::new(DescriptorAlloc::new());

        Self {
            inner: Arc::new(Inner {
                logical,
                physical,
                properties,
                features,
                allocator,
                descriptors,
                samplers_cache: Default::default(),
                epochs: Epochs::new(queues),
            }),
        }
    }

    pub(crate) fn epochs(&self) -> &Epochs {
        &self.inner.epochs
    }

    pub fn graphics(&self) -> &'static Graphics {
        unsafe { Graphics::get_unchecked() }
    }

    pub fn logical(&self) -> &vulkanalia::Device {
        &self.inner.logical
    }

    pub fn physical(&self) -> vk::PhysicalDevice {
        self.inner.physical
    }

    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.inner.properties.v1_0.limits
    }

    pub fn properties(&self) -> &DeviceProperties {
        &self.inner.properties
    }

    pub fn features(&self) -> &DeviceFeatures {
        &self.inner.features
    }

    pub fn downgrade(&self) -> WeakDevice {
        WeakDevice(Arc::downgrade(&self.inner))
    }

    pub fn wait_idle(&self) -> Result<(), DeviceLost> {
        self.inner.wait_idle()
    }

    pub fn map_memory(
        &self,
        buffer: &mut MappableBuffer,
        offset: u64,
        size: usize,
    ) -> Result<&mut [MaybeUninit<u8>], MapError> {
        Ok(unsafe {
            let ptr = buffer
                .memory_block()
                .map(self.logical().as_memory_device(), offset, size)?;

            std::slice::from_raw_parts_mut(ptr.as_ptr() as _, size)
        })
    }

    pub fn unmap_memory(&self, buffer: &mut MappableBuffer) {
        unsafe {
            buffer
                .memory_block()
                .unmap(self.logical().as_memory_device());
        }
    }

    pub fn upload_to_memory<T>(
        &self,
        buffer: &mut MappableBuffer,
        offset: u64,
        data: &[T],
    ) -> Result<(), MapError>
    where
        T: bytemuck::Pod,
    {
        let slice = self.map_memory(buffer, offset, data.len())?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                slice.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(data),
            )
        }

        self.unmap_memory(buffer);
        Ok(())
    }

    pub fn create_semaphore(&self) -> Result<Semaphore, OutOfDeviceMemory> {
        let logical = &self.inner.logical;

        let info = vk::SemaphoreCreateInfo::builder();
        let handle = unsafe { logical.create_semaphore(&info, None) }
            .map_err(OutOfDeviceMemory::on_creation)?;

        tracing::debug!(semaphore = ?handle, "created semaphore");

        Ok(Semaphore::new(handle, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_semaphore(&self, handle: vk::Semaphore) {
        self.logical().destroy_semaphore(handle, None);
    }

    pub fn create_fence(&self) -> Result<Fence, OutOfDeviceMemory> {
        let logical = &self.inner.logical;

        let info = vk::FenceCreateInfo::builder();
        let handle =
            unsafe { logical.create_fence(&info, None) }.map_err(OutOfDeviceMemory::on_creation)?;

        tracing::debug!(fence = ?handle, "created fence");

        Ok(Fence::new(handle, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_fence(&self, handle: vk::Fence) {
        self.logical().destroy_fence(handle, None);
    }

    pub fn update_armed_fence_state(&self, fence: &mut Fence) -> Result<bool, DeviceLost> {
        let status =
            unsafe { self.logical().get_fence_status(fence.handle()) }.map_err(|e| match e {
                vk::ErrorCode::DEVICE_LOST => DeviceLost,
                vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
                _ => crate::unexpected_vulkan_error(e),
            })?;

        match status {
            vk::SuccessCode::SUCCESS => {
                if let Some((queue, epoch)) = fence.set_signalled() {
                    self.epochs().close_epoch(queue, epoch);
                }
                Ok(true)
            }
            vk::SuccessCode::NOT_READY => Ok(false),
            c => panic!("unexpected status code: {c:?}"),
        }
    }

    pub fn reset_fences(&self, fences: &mut [&mut Fence]) -> Result<(), DeviceLost> {
        let handles = fences
            .iter_mut()
            .map(|fence| {
                if matches!(fence.state(), FenceState::Armed { .. }) {
                    let signalled = self.update_armed_fence_state(fence)?;

                    // Armed and not signalled yet -> logic error
                    assert!(signalled, "armed fence cannot be reset");
                }
                Ok(fence.handle())
            })
            .collect::<Result<SmallVec<[_; 16]>, DeviceLost>>()?;

        if let Err(e) = unsafe { self.logical().reset_fences(&handles) } {
            crate::unexpected_vulkan_error(e);
        }

        for fence in fences {
            fence.set_unsignalled();
        }

        Ok(())
    }

    pub fn wait_fences(&self, fences: &mut [&mut Fence], wait_all: bool) -> Result<(), DeviceLost> {
        let handles = fences
            .iter()
            .filter_map(|fence| match fence.state() {
                // Waiting for an unarmed fence -> error (preventing deadlock)
                FenceState::Unsignalled => {
                    // Logic error
                    panic!("waiting for an unarmed fence")
                }
                // Waiting for an armed fence -> ok
                FenceState::Armed { .. } => Some(fence.handle()),
                // Already signalled fences could be skipped
                FenceState::Signalled => None,
            })
            .collect::<SmallVec<[_; 16]>>();

        if handles.is_empty() {
            return Ok(());
        }

        unsafe {
            self.inner
                .logical
                .wait_for_fences(&handles, wait_all, u64::MAX)
        }
        .map_err(|e| match e {
            vk::ErrorCode::DEVICE_LOST => DeviceLost,
            vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
            _ => crate::unexpected_vulkan_error(e),
        })?;

        let all_signalled = wait_all || handles.len() == 1;

        let mut epochs_to_close = SmallVec::<[_; 16]>::new();

        for fence in fences {
            if all_signalled || self.update_armed_fence_state(fence)? {
                if let Some(epoch) = fence.set_signalled() {
                    epochs_to_close.push(epoch);
                }
            }
        }

        if !epochs_to_close.is_empty() {
            epochs_to_close.sort_unstable_by_key(|(q, e)| (*q, std::cmp::Reverse(*e)));
            let mut last_queue = None;
            epochs_to_close.retain(|(q, _)| {
                if last_queue == Some(*q) {
                    false
                } else {
                    last_queue = Some(*q);
                    true
                }
            });

            for (queue, epoch) in epochs_to_close {
                self.epochs().close_epoch(queue, epoch);
            }
        }

        Ok(())
    }

    pub fn create_surface(&self, window: Arc<Window>) -> Result<Surface, CreateSurfaceError> {
        let surface = Surface::new(self.graphics().instance(), window, self)?;

        tracing::debug!(surface = ?surface.handle(), "created surface");
        Ok(surface)
    }

    pub fn create_buffer(&self, info: BufferInfo) -> Result<Buffer, OutOfDeviceMemory> {
        self.create_buffer_impl(info, None)
            .map(MappableBuffer::freeze)
    }

    pub fn create_mappable_buffer(
        &self,
        info: BufferInfo,
        memory_usage: MemoryUsage,
    ) -> Result<MappableBuffer, OutOfDeviceMemory> {
        self.create_buffer_impl(info, Some(memory_usage))
    }

    fn create_buffer_impl(
        &self,
        info: BufferInfo,
        memory_usage: Option<MemoryUsage>,
    ) -> Result<MappableBuffer, OutOfDeviceMemory> {
        let logical = &self.inner.logical;

        let mut alloc_flags = gpu_alloc::UsageFlags::empty();
        if let Some(memory_usage) = memory_usage {
            // NOTE: memory usage is passed for the mappable buffer only.
            alloc_flags |= gpu_alloc::UsageFlags::HOST_ACCESS;
            if memory_usage.contains(MemoryUsage::UPLOAD) {
                alloc_flags |= gpu_alloc::UsageFlags::UPLOAD;
            }
            if memory_usage.contains(MemoryUsage::DOWNLOAD) {
                alloc_flags |= gpu_alloc::UsageFlags::DOWNLOAD;
            }
            if memory_usage.contains(MemoryUsage::FAST_DEVICE_ACCESS) {
                alloc_flags |= gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS;
            }
            if memory_usage.contains(MemoryUsage::TRANSIENT) {
                alloc_flags |= gpu_alloc::UsageFlags::TRANSIENT;
            }
        }

        let has_device_address = info.usage.contains(BufferUsage::SHADER_DEVICE_ADDRESS);
        if has_device_address {
            assert!(
                self.inner.features.v1_2.buffer_device_address != 0,
                "`SHADER_DEVICE_ADDRESS` buffer usage requires `BufferDeviceAddress`
                feature"
            );
            alloc_flags |= gpu_alloc::UsageFlags::DEVICE_ADDRESS;
        }

        let handle = {
            let info = vk::BufferCreateInfo::builder()
                .size(info.size)
                .usage(info.usage.to_vk())
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            unsafe { logical.create_buffer(&info, None) }.map_err(OutOfDeviceMemory::on_creation)?
        }
        .with_defer(|handle| unsafe { logical.destroy_buffer(handle, None) });

        let mut dedicated = vk::MemoryDedicatedRequirements::builder();
        let mut reqs = vk::MemoryRequirements2::builder().push_next(&mut dedicated);
        if self.graphics().vk1_1() {
            let info = vk::BufferMemoryRequirementsInfo2::builder().buffer(*handle);
            unsafe { logical.get_buffer_memory_requirements2(&info, &mut reqs) }
        } else {
            reqs.memory_requirements = unsafe { logical.get_buffer_memory_requirements(*handle) };
        }

        debug_assert!(reqs.memory_requirements.alignment.is_power_of_two());

        let block = {
            let request = gpu_alloc::Request {
                size: reqs.memory_requirements.size,
                align_mask: (reqs.memory_requirements.alignment - 1) | info.align,
                usage: alloc_flags,
                memory_types: reqs.memory_requirements.memory_type_bits,
            };

            let dedicated = if dedicated.requires_dedicated_allocation != 0 {
                Some(gpu_alloc::Dedicated::Required)
            } else if dedicated.prefers_dedicated_allocation != 0 {
                Some(gpu_alloc::Dedicated::Preferred)
            } else {
                None
            };

            let logical = logical.as_memory_device();
            let mut allocator = self.inner.allocator.lock().unwrap();
            unsafe {
                match dedicated {
                    None => allocator.alloc(logical, request),
                    Some(dedicated) => allocator.alloc_with_dedicated(logical, request, dedicated),
                }
            }
            .map_err(|e| match e {
                gpu_alloc::AllocationError::OutOfDeviceMemory => OutOfDeviceMemory,
                gpu_alloc::AllocationError::OutOfHostMemory => crate::out_of_host_memory(),
                _ => panic!("unexpected allocation error: {e:?}"),
            })?
        };

        unsafe { logical.bind_buffer_memory(*handle, *block.memory(), block.offset()) }
            .map_err(OutOfDeviceMemory::on_creation)?;

        let address = if has_device_address {
            let info = vk::BufferDeviceAddressInfo::builder().buffer(*handle);
            let address = unsafe { logical.get_buffer_device_address(&info) };
            Some(DeviceAddress::new(address).unwrap())
        } else {
            None
        };

        tracing::debug!(buffer = ?*handle, "created buffer");

        Ok(MappableBuffer::new(
            handle.disarm(),
            info,
            alloc_flags,
            address,
            self.downgrade(),
            block,
        ))
    }

    pub(crate) unsafe fn destroy_buffer(
        &self,
        handle: vk::Buffer,
        block: MemoryBlock<vk::DeviceMemory>,
    ) {
        self.inner
            .allocator
            .lock()
            .unwrap()
            .dealloc(self.logical().as_memory_device(), block);

        self.logical().destroy_buffer(handle, None);
    }

    pub fn create_buffer_view(
        &self,
        info: BufferViewInfo,
    ) -> Result<BufferView, OutOfDeviceMemory> {
        assert!(
            info.buffer
                .info()
                .usage
                .contains(BufferUsage::UNIFORM_TEXEL | BufferUsage::STORAGE_TEXEL),
            "buffer view cannot be created from a buffer without at least one of \
            `UNIFORM_TEXEL` or `STORAGE_TEXEL` usages"
        );

        let logical = &self.inner.logical;

        let handle = {
            let info = vk::BufferViewCreateInfo::builder()
                .buffer(info.buffer.handle())
                .format(info.format.to_vk())
                .offset(info.offset)
                .range(info.size);

            unsafe { logical.create_buffer_view(&info, None) }
                .map_err(OutOfDeviceMemory::on_creation)?
        };

        tracing::debug!(buffer_view = ?handle, "created buffer view");

        Ok(BufferView::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_buffer_view(&self, handle: vk::BufferView) {
        self.logical().destroy_buffer_view(handle, None);
    }

    pub fn create_image(&self, info: ImageInfo) -> Result<Image, OutOfDeviceMemory> {
        let logical = &self.inner.logical;

        let handle = {
            let info = vk::ImageCreateInfo::builder()
                .image_type(info.extent.to_vk())
                .format(info.format.to_vk())
                .extent(vk::Extent3D::from_gfx(info.extent))
                .mip_levels(info.mip_levels)
                .samples(info.samples.to_vk())
                .array_layers(info.array_layers)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(info.usage.to_vk())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            // NOTE: `INVALID_OPAQUE_CAPTURE_ADDRESS` might be returned here, but
            // we cannot handle it anyway.
            unsafe { logical.create_image(&info, None) }.map_err(OutOfDeviceMemory::on_creation)?
        }
        .with_defer(|image| unsafe { logical.destroy_image(image, None) });

        let mut dedicated = vk::MemoryDedicatedRequirements::builder();
        let mut reqs = vk::MemoryRequirements2::builder().push_next(&mut dedicated);
        if self.graphics().vk1_1() {
            let info = vk::ImageMemoryRequirementsInfo2::builder().image(*handle);
            unsafe { logical.get_image_memory_requirements2(&info, &mut reqs) }
        } else {
            reqs.memory_requirements = unsafe { logical.get_image_memory_requirements(*handle) };
        }

        debug_assert!(reqs.memory_requirements.alignment.is_power_of_two());

        let block = {
            let request = gpu_alloc::Request {
                size: reqs.memory_requirements.size,
                align_mask: reqs.memory_requirements.alignment - 1,
                usage: gpu_alloc::UsageFlags::empty(),
                memory_types: reqs.memory_requirements.memory_type_bits,
            };

            let dedicated = if dedicated.requires_dedicated_allocation != 0 {
                Some(gpu_alloc::Dedicated::Required)
            } else if dedicated.prefers_dedicated_allocation != 0 {
                Some(gpu_alloc::Dedicated::Preferred)
            } else {
                None
            };

            let logical = logical.as_memory_device();
            let mut allocator = self.inner.allocator.lock().unwrap();
            unsafe {
                match dedicated {
                    None => allocator.alloc(logical, request),
                    Some(dedicated) => allocator.alloc_with_dedicated(logical, request, dedicated),
                }
            }
        }
        .map_err(|e| match e {
            gpu_alloc::AllocationError::OutOfDeviceMemory => OutOfDeviceMemory,
            gpu_alloc::AllocationError::OutOfHostMemory => crate::out_of_host_memory(),
            _ => panic!("unexpected allocation error: {e:?}"),
        })?;

        unsafe { logical.bind_image_memory(*handle, *block.memory(), block.offset()) }
            .map_err(OutOfDeviceMemory::on_creation)?;

        tracing::debug!(image = ?*handle, "created image");

        Ok(Image::new(handle.disarm(), info, self.downgrade(), block))
    }

    pub(crate) unsafe fn destroy_image(
        &self,
        handle: vk::Image,
        block: MemoryBlock<vk::DeviceMemory>,
    ) {
        self.inner
            .allocator
            .lock()
            .unwrap()
            .dealloc(self.logical().as_memory_device(), block);

        self.logical().destroy_image(handle, None)
    }

    pub fn create_image_view(&self, info: ImageViewInfo) -> Result<ImageView, OutOfDeviceMemory> {
        let logical = &self.inner.logical;

        let handle = {
            let info = vk::ImageViewCreateInfo::builder()
                .image(info.image.handle())
                .format(info.image.info().format.to_vk())
                .view_type(info.ty.to_vk())
                .subresource_range(vk::ImageSubresourceRange::from_gfx(info.range))
                .components(vk::ComponentMapping::from_gfx(info.mapping));

            // NOTE: `INVALID_OPAQUE_CAPTURE_ADDRESS` might be returned here, but
            // we cannot handle it anyway.
            unsafe { logical.create_image_view(&info, None) }
                .map_err(OutOfDeviceMemory::on_creation)?
        };

        tracing::debug!(image_view = ?handle, "created image view");

        Ok(ImageView::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_image_view(&self, handle: vk::ImageView) {
        self.logical().destroy_image_view(handle, None);
    }

    pub fn create_sampler(&self, info: SamplerInfo) -> Result<Sampler, OutOfDeviceMemory> {
        use dashmap::mapref::entry::Entry;

        let logical = &self.inner.logical;

        let sampler = match self.inner.samplers_cache.entry(info) {
            Entry::Occupied(entry) => {
                return Ok(entry.get().clone());
            }
            Entry::Vacant(entry) => {
                let handle = {
                    let mut create_info = vk::SamplerCreateInfo::builder()
                        .mag_filter(info.mag_filter.to_vk())
                        .min_filter(info.min_filter.to_vk())
                        .mipmap_mode(info.mipmap_mode.to_vk())
                        .address_mode_u(info.address_mode_u.to_vk())
                        .address_mode_v(info.address_mode_v.to_vk())
                        .address_mode_w(info.address_mode_w.to_vk())
                        .mip_lod_bias(info.mip_lod_bias)
                        .anisotropy_enable(info.max_anisotropy.is_some())
                        .max_anisotropy(info.max_anisotropy.unwrap_or_default())
                        .compare_enable(info.compare_op.is_some())
                        .compare_op(info.compare_op.to_vk())
                        .min_lod(info.min_lod)
                        .max_lod(info.max_lod)
                        .border_color(info.border_color.to_vk())
                        .unnormalized_coordinates(info.unnormalized_coordinates);

                    let mut reduction_mode_info;
                    if let Some(reduction_mode) = info.reduction_mode {
                        reduction_mode_info = vk::SamplerReductionModeCreateInfo::builder()
                            .reduction_mode(reduction_mode.to_vk());
                        create_info = create_info.push_next(&mut reduction_mode_info);
                    }

                    // NOTE: `INVALID_OPAQUE_CAPTURE_ADDRESS` might be returned here, but
                    // we cannot handle it anyway.
                    unsafe { logical.create_sampler(&create_info, None) }
                        .map_err(OutOfDeviceMemory::on_creation)?
                };

                entry
                    .insert(Sampler::new(handle, info, self.downgrade()))
                    .clone()
            }
        };

        tracing::debug!(sampler = ?sampler.handle(), "created sampler");

        Ok(sampler)
    }

    pub(crate) unsafe fn destroy_sampler(&self, handle: vk::Sampler) {
        self.logical().destroy_sampler(handle, None)
    }

    pub fn create_shader_module(
        &self,
        info: ShaderModuleInfo,
    ) -> Result<ShaderModule, OutOfDeviceMemory> {
        let handle = {
            let info = vk::ShaderModuleCreateInfo::builder()
                .code_size(info.data.len() * 4)
                .code(&info.data);

            // NOTE: `INVALID_SHADER_NV` is not possible as soon as `VK_NV_glsl_shader` extension
            // is not enabled. (it won't, because we use SPIR-V)
            unsafe { self.logical().create_shader_module(&info, None) }
                .map_err(OutOfDeviceMemory::on_creation)?
        };

        tracing::debug!(shader_module = ?handle, "created shader module");

        Ok(ShaderModule::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_shader_module(&self, handle: vk::ShaderModule) {
        self.logical().destroy_shader_module(handle, None);
    }

    pub fn create_render_pass(
        &self,
        info: RenderPassInfo,
    ) -> Result<RenderPass, CreateRenderPassError> {
        let mut subpass_attachments = Vec::new();

        let mut subpasses = SmallVec::<[_; 4]>::with_capacity(info.subpasses.len());
        for (subpass_index, subpass) in info.subpasses.iter().enumerate() {
            let color_offset = subpass_attachments.len();
            subpass_attachments.reserve(subpass.colors.len() + subpass.depth.is_some() as usize);

            for (color_index, &(i, layout)) in subpass.colors.iter().enumerate() {
                if i as usize >= info.attachments.len() {
                    return Err(CreateRenderPassError::ColorAttachmentOutOfBounds {
                        attachment_index: i,
                        color_index,
                        subpass_index,
                    });
                }

                subpass_attachments.push(
                    vk::AttachmentReference::builder()
                        .attachment(i)
                        .layout(layout.to_vk()),
                );
            }

            let depths_offset = subpass_attachments.len();
            if let Some((i, layout)) = subpass.depth {
                if i as usize >= info.attachments.len() {
                    return Err(CreateRenderPassError::DepthAttachmentOutOfBounds {
                        attachment_index: i,
                        subpass_index,
                    });
                }

                subpass_attachments.push(
                    vk::AttachmentReference::builder()
                        .attachment(i)
                        .layout(layout.to_vk()),
                );
            }

            subpasses.push((color_offset, depths_offset));
        }
        let subpasses = info
            .subpasses
            .iter()
            .zip(subpasses)
            .map(|(subpass, (color_offset, depths_offset))| {
                let descr = vk::SubpassDescription::builder()
                    .color_attachments(&subpass_attachments[color_offset..depths_offset]);
                if subpass.depth.is_some() {
                    descr.depth_stencil_attachment(&subpass_attachments[depths_offset])
                } else {
                    descr
                }
            })
            .collect::<Vec<_>>();

        let attachments = info
            .attachments
            .iter()
            .map(|info| {
                vk::AttachmentDescription::builder()
                    .format(info.format.to_vk())
                    .load_op(info.load_op.to_vk())
                    .store_op(info.store_op.to_vk())
                    .initial_layout(info.initial_layout.to_vk())
                    .final_layout(info.final_layout.to_vk())
                    .samples(vk::SampleCountFlags::_1)
            })
            .collect::<Vec<_>>();

        let dependencies = info
            .dependencies
            .iter()
            .map(|dep| vk::SubpassDependency::from_gfx(*dep))
            .collect::<Vec<_>>();

        let handle = {
            let info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies);

            unsafe { self.logical().create_render_pass(&info, None) }
                .map_err(OutOfDeviceMemory::on_creation)?
        };

        tracing::debug!(render_pass = ?handle, "created render pass");

        Ok(RenderPass::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_render_pass(&self, handle: vk::RenderPass) {
        self.logical().destroy_render_pass(handle, None);
    }

    pub fn create_framebuffer(
        &self,
        info: FramebufferInfo,
    ) -> Result<Framebuffer, OutOfDeviceMemory> {
        assert!(
            info.attachments
                .iter()
                .all(|view| view.info().ty == ImageViewType::D2),
            "all image views must be 2d images"
        );

        assert!(
            info.attachments.iter().all(|view| {
                let extent: vk::Extent2D = view.info().image.info().extent.to_vk();
                extent.width >= info.extent.x && extent.height >= info.extent.y
            }),
            "all image views must have at least the framebuffer extent"
        );

        let render_pass = info.render_pass.handle();
        let attachments = info
            .attachments
            .iter()
            .map(ImageView::handle)
            .collect::<SmallVec<[_; 8]>>();

        let handle = {
            let info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(info.extent.x)
                .height(info.extent.y)
                .layers(1);

            unsafe { self.logical().create_framebuffer(&info, None) }
                .map_err(OutOfDeviceMemory::on_creation)?
        };

        tracing::debug!(framebuffer = ?handle, "created framebuffer");

        Ok(Framebuffer::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_framebuffer(&self, handle: vk::Framebuffer) {
        self.logical().destroy_framebuffer(handle, None);
    }

    pub fn create_descriptor_set_layout(
        &self,
        info: DescriptorSetLayoutInfo,
    ) -> Result<DescriptorSetLayout, OutOfDeviceMemory> {
        let graphics = self.graphics();
        let logical = &self.inner.logical;

        let handle = {
            let flags;
            let mut flags_info;

            let mut create_info =
                vk::DescriptorSetLayoutCreateInfo::builder().flags(info.flags.to_vk());

            if graphics.vk1_2() || self.features().v1_2.descriptor_indexing != 0 {
                if info
                    .bindings
                    .iter()
                    .any(|b| b.flags.contains(DescriptorBindingFlags::UPDATE_AFTER_BIND))
                {
                    assert!(
                        info.flags
                            .contains(DescriptorSetLayoutFlags::UPDATE_AFTER_BIND_POOL),
                        "`UPDATE_AFTER_BIND_POOL` flag must be set in descriptor set layout \
                        create info flags to use `UPDATE_AFTER_BIND` binding flags"
                    );
                }

                flags = info
                    .bindings
                    .iter()
                    .map(|b| b.flags.to_vk())
                    .collect::<SmallVec<[_; 8]>>();

                flags_info =
                    vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);
                create_info = create_info.push_next(&mut flags_info);
            } else {
                assert!(
                    info.bindings.iter().all(|b| b.flags.is_empty()),
                    "Vulkan 1.2 or descriptor indexing extension are required \
                    for non-empty `DescriptorBindingFlags`"
                );
            }

            let bindings = info
                .bindings
                .iter()
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(binding.binding)
                        .descriptor_count(binding.count)
                        .descriptor_type(binding.ty.to_vk())
                        .stage_flags(binding.stages.to_vk())
                })
                .collect::<SmallVec<[_; 8]>>();

            create_info = create_info.bindings(&bindings);

            unsafe { logical.create_descriptor_set_layout(&create_info, None) }
                .map_err(OutOfDeviceMemory::on_creation)?
        };

        tracing::debug!(descriptor_set_layout = ?handle, "created descriptor set layout");

        let mut size = DescriptorSetSize::default();
        for binding in info.bindings.iter() {
            match binding.ty {
                DescriptorType::Sampler => size.samplers += binding.count,
                DescriptorType::CombinedImageSampler => {
                    size.combined_image_samplers += binding.count
                }
                DescriptorType::SampledImage => size.sampled_images += binding.count,
                DescriptorType::StorageImage => size.storage_images += binding.count,
                DescriptorType::UniformTexelBuffer => size.uniform_texel_buffers += binding.count,
                DescriptorType::StorageTexelBuffer => size.storage_texel_buffers += binding.count,
                DescriptorType::UniformBuffer => size.uniform_buffers += binding.count,
                DescriptorType::StorageBuffer => size.storage_buffers += binding.count,
                DescriptorType::UniformBufferDynamic => {
                    size.uniform_buffers_dynamic += binding.count
                }
                DescriptorType::StorageBufferDynamic => {
                    size.storage_buffers_dynamic += binding.count
                }
                DescriptorType::InputAttachment => size.input_attachments += binding.count,
            }
        }

        Ok(DescriptorSetLayout::new(
            handle,
            info,
            size,
            self.downgrade(),
        ))
    }

    pub(crate) unsafe fn destroy_descriptor_set_layout(&self, handle: vk::DescriptorSetLayout) {
        self.logical().destroy_descriptor_set_layout(handle, None)
    }

    pub fn create_descriptor_set(
        &self,
        info: DescriptorSetInfo,
    ) -> Result<WritableDescriptorSet, DescriptorAllocError> {
        assert!(
            !info
                .layout
                .info()
                .flags
                .contains(DescriptorSetLayoutFlags::PUSH_DESCRIPTOR),
            "push descriptor sets cannot be created"
        );

        let set = {
            let mut descriptors = self.inner.descriptors.lock().unwrap();
            let mut sets = unsafe { descriptors.allocate(self.logical(), &info.layout, 1) }?;
            sets.remove(0)
        };

        tracing::debug!(descriptor_set = ?set.handle(), "created descriptor set");

        Ok(WritableDescriptorSet::new(set, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_descriptor_set(&self, allocated: &AllocatedDescriptorSet) {
        self.inner
            .descriptors
            .lock()
            .unwrap()
            .free(self.logical(), std::slice::from_ref(allocated))
    }

    pub fn update_descriptor_sets(&self, updates: &mut [UpdateDescriptorSet<'_>]) {
        struct UpdatesIter<I> {
            inner: I,
            len: usize,
        }

        impl<T, I: Iterator<Item = T>> Iterator for UpdatesIter<I> {
            type Item = T;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        impl<I: Iterator> ExactSizeIterator for UpdatesIter<I> {
            #[inline]
            fn len(&self) -> usize {
                self.len
            }
        }

        let alloc = Bump::new();

        let writes = {
            let len = updates.iter().map(|update| update.writes.len()).sum();
            alloc.alloc_slice_fill_iter(UpdatesIter {
                inner: updates.iter().flat_map(|update| {
                    update.writes.iter().map(|write| {
                        vk::WriteDescriptorSet::builder()
                            .dst_set(update.set.handle())
                            .dst_binding(write.binding)
                            .dst_array_element(write.element)
                            .build()
                    })
                }),
                len,
            })
        };

        let mut writes_iter = writes.iter_mut();
        for update in updates.iter() {
            for write in update.writes.iter() {
                let descr = writes_iter.next().unwrap();

                match write.data {
                    DescriptorSlice::Sampler(data) => {
                        let images = alloc.alloc_slice_fill_iter(data.iter().map(|sampler| {
                            vk::DescriptorImageInfo::builder().sampler(sampler.handle())
                        }));
                        descr.descriptor_type = vk::DescriptorType::SAMPLER;
                        descr.descriptor_count = images.len() as _;
                        descr.image_info = images.as_ptr().cast();
                    }
                    DescriptorSlice::CombinedImageSampler(data) => {
                        let images = alloc.alloc_slice_fill_iter(data.iter().map(|item| {
                            vk::DescriptorImageInfo::builder()
                                .sampler(item.sampler.handle())
                                .image_view(item.view.handle())
                                .image_layout(item.layout.to_vk())
                        }));
                        descr.descriptor_type = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
                        descr.descriptor_count = images.len() as _;
                        descr.image_info = images.as_ptr().cast();
                    }
                    DescriptorSlice::SampledImage(data) => {
                        let images =
                            alloc.alloc_slice_fill_iter(data.iter().map(|(view, layout)| {
                                vk::DescriptorImageInfo::builder()
                                    .image_view(view.handle())
                                    .image_layout((*layout).to_vk())
                            }));
                        descr.descriptor_type = vk::DescriptorType::SAMPLED_IMAGE;
                        descr.descriptor_count = images.len() as _;
                        descr.image_info = images.as_ptr().cast();
                    }
                    DescriptorSlice::StorageImage(data) => {
                        let images =
                            alloc.alloc_slice_fill_iter(data.iter().map(|(view, layout)| {
                                vk::DescriptorImageInfo::builder()
                                    .image_view(view.handle())
                                    .image_layout((*layout).to_vk())
                            }));
                        descr.descriptor_type = vk::DescriptorType::STORAGE_IMAGE;
                        descr.descriptor_count = images.len() as _;
                        descr.image_info = images.as_ptr().cast();
                    }
                    DescriptorSlice::UniformTexelBuffer(data) => {
                        let views =
                            alloc.alloc_slice_fill_iter(data.iter().map(BufferView::handle));
                        descr.descriptor_type = vk::DescriptorType::UNIFORM_TEXEL_BUFFER;
                        descr.descriptor_count = views.len() as _;
                        descr.texel_buffer_view = views.as_ptr().cast();
                    }
                    DescriptorSlice::StorageTexelBuffer(data) => {
                        let views =
                            alloc.alloc_slice_fill_iter(data.iter().map(BufferView::handle));
                        descr.descriptor_type = vk::DescriptorType::STORAGE_TEXEL_BUFFER;
                        descr.descriptor_count = views.len() as _;
                        descr.texel_buffer_view = views.as_ptr().cast();
                    }
                    DescriptorSlice::UniformBuffer(data) => {
                        let buffers = alloc.alloc_slice_fill_iter(data.iter().map(|range| {
                            vk::DescriptorBufferInfo::builder()
                                .buffer(range.buffer.handle())
                                .offset(range.offset)
                                .range(range.size)
                        }));
                        descr.descriptor_type = vk::DescriptorType::UNIFORM_BUFFER;
                        descr.descriptor_count = buffers.len() as _;
                        descr.buffer_info = buffers.as_ptr().cast();
                    }
                    DescriptorSlice::StorageBuffer(data) => {
                        let buffers = alloc.alloc_slice_fill_iter(data.iter().map(|range| {
                            vk::DescriptorBufferInfo::builder()
                                .buffer(range.buffer.handle())
                                .offset(range.offset)
                                .range(range.size)
                        }));
                        descr.descriptor_type = vk::DescriptorType::STORAGE_BUFFER;
                        descr.descriptor_count = buffers.len() as _;
                        descr.buffer_info = buffers.as_ptr().cast();
                    }
                    DescriptorSlice::UniformBufferDynamic(data) => {
                        let buffers = alloc.alloc_slice_fill_iter(data.iter().map(|range| {
                            vk::DescriptorBufferInfo::builder()
                                .buffer(range.buffer.handle())
                                .offset(range.offset)
                                .range(range.size)
                        }));
                        descr.descriptor_type = vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC;
                        descr.descriptor_count = buffers.len() as _;
                        descr.buffer_info = buffers.as_ptr().cast();
                    }
                    DescriptorSlice::StorageBufferDynamic(data) => {
                        let buffers = alloc.alloc_slice_fill_iter(data.iter().map(|range| {
                            vk::DescriptorBufferInfo::builder()
                                .buffer(range.buffer.handle())
                                .offset(range.offset)
                                .range(range.size)
                        }));
                        descr.descriptor_type = vk::DescriptorType::STORAGE_BUFFER_DYNAMIC;
                        descr.descriptor_count = buffers.len() as _;
                        descr.buffer_info = buffers.as_ptr().cast();
                    }
                    DescriptorSlice::InputAttachment(data) => {
                        let images =
                            alloc.alloc_slice_fill_iter(data.iter().map(|(view, layout)| {
                                vk::DescriptorImageInfo::builder()
                                    .image_view(view.handle())
                                    .image_layout((*layout).to_vk())
                            }));
                        descr.descriptor_type = vk::DescriptorType::INPUT_ATTACHMENT;
                        descr.descriptor_count = images.len() as _;
                        descr.image_info = images.as_ptr().cast();
                    }
                }
            }
        }
        debug_assert!(writes_iter.next().is_none());

        for update in updates {
            for write in update.writes {
                update
                    .set
                    .write_descriptors(write.binding, write.element, write.data);
            }
        }

        unsafe {
            self.logical()
                .update_descriptor_sets(writes, &([] as [vk::CopyDescriptorSet; 0]))
        };
    }

    pub fn create_pipeline_layout(
        &self,
        info: PipelineLayoutInfo,
    ) -> Result<PipelineLayout, OutOfDeviceMemory> {
        let logical = &self.inner.logical;

        let handle = {
            let sets = info
                .sets
                .iter()
                .map(|set| set.handle())
                .collect::<SmallVec<[_; 8]>>();
            let push_constants = info
                .push_constants
                .iter()
                .map(|c| vk::PushConstantRange::from_gfx(*c))
                .collect::<SmallVec<[_; 8]>>();

            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&sets)
                .push_constant_ranges(&push_constants);

            unsafe { logical.create_pipeline_layout(&info, None) }
                .map_err(OutOfDeviceMemory::on_creation)?
        };

        tracing::debug!(pipeline_layout = ?handle, "created pipeline layout");

        Ok(PipelineLayout::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_pipeline_layout(&self, handle: vk::PipelineLayout) {
        self.logical().destroy_pipeline_layout(handle, None)
    }

    pub fn create_graphics_pipeline(
        &self,
        info: GraphicsPipelineInfo,
    ) -> Result<GraphicsPipeline, OutOfDeviceMemory> {
        let logical = &self.inner.logical;
        let descr = &info.descr;

        let mut create_info = vk::GraphicsPipelineCreateInfo::builder();

        let color_count = {
            let r = &info.rendering;

            let subpass = r
                .render_pass
                .info()
                .subpasses
                .get(r.subpass as usize)
                .expect("subpass index is out of bounds");

            create_info = create_info
                .render_pass(r.render_pass.handle())
                .subpass(r.subpass);

            subpass.colors.len()
        };

        let mut shader_stages = Vec::with_capacity(2);

        // Vertex input state
        let vertex_binding_descriptions = descr
            .vertex_bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                vk::VertexInputBindingDescription::builder()
                    .binding(i as u32)
                    .stride(b.stride)
                    .input_rate(b.rate.to_vk())
            })
            .collect::<SmallVec<[_; 4]>>();

        let vertex_attribute_descriptions = descr
            .vertex_attributes
            .iter()
            .map(|a| vk::VertexInputAttributeDescription::from_gfx(*a))
            .collect::<SmallVec<[_; 8]>>();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        let vertex_shader_entry =
            vk::StringArray::<64>::from_bytes(descr.vertex_shader.entry().as_bytes());

        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(descr.vertex_shader.module().handle())
                .name(vertex_shader_entry.as_bytes()),
        );

        // Input assembly state
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(descr.primitive_topology.to_vk())
            .primitive_restart_enable(descr.primitive_restart_enable);

        // Rasterizer
        let fragment_shader_entry;
        let attachments;
        let mut viewport_state = vk::PipelineViewportStateCreateInfo::builder();
        let mut multisample_state = vk::PipelineMultisampleStateCreateInfo::builder();
        let mut depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder();
        let mut color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder();

        let mut dynamic_states = Vec::with_capacity(7);
        let rasterization_state = match &descr.rasterizer {
            Some(rasterizer) => {
                // Viewport and scissors state
                match &rasterizer.viewport {
                    State::Static(viewport) => {
                        viewport_state = viewport_state.viewports(std::slice::from_ref(viewport));
                    }
                    State::Dynamic => {
                        dynamic_states.push(vk::DynamicState::VIEWPORT);
                        viewport_state = viewport_state.viewport_count(1);
                    }
                }
                match &rasterizer.scissor {
                    State::Static(scissor) => {
                        viewport_state = viewport_state.scissors(std::slice::from_ref(scissor));
                    }
                    State::Dynamic => {
                        dynamic_states.push(vk::DynamicState::SCISSOR);
                        viewport_state = viewport_state.scissor_count(1);
                    }
                }

                // Multisample state
                multisample_state =
                    multisample_state.rasterization_samples(vk::SampleCountFlags::_1);

                // Depth/stencil state
                if let Some(depth_test) = rasterizer.depth_test {
                    depth_stencil_state = depth_stencil_state
                        .depth_test_enable(true)
                        .depth_write_enable(depth_test.write)
                        .depth_compare_op(depth_test.compare.to_vk())
                }
                if let Some(depth_bounds) = rasterizer.depth_bounds {
                    depth_stencil_state = depth_stencil_state.depth_bounds_test_enable(true);

                    match depth_bounds {
                        State::Static(bounds) => {
                            depth_stencil_state = depth_stencil_state
                                .min_depth_bounds(bounds.offset)
                                .max_depth_bounds(bounds.offset + bounds.size);
                        }
                        State::Dynamic => {
                            dynamic_states.push(vk::DynamicState::DEPTH_BOUNDS);
                        }
                    }
                }
                if let Some(stencil_tests) = &rasterizer.stencil_tests {
                    fn make_stencil_test(
                        test: &StencilTest,
                        dynamic_states: &mut Vec<vk::DynamicState>,
                    ) -> vk::StencilOpStateBuilder {
                        let mut builder = vk::StencilOpState::builder()
                            .fail_op(test.fail.to_vk())
                            .pass_op(test.pass.to_vk())
                            .depth_fail_op(test.depth_fail.to_vk())
                            .compare_op(test.compare.to_vk());

                        match test.compare_mask {
                            State::Static(mask) => builder = builder.compare_mask(mask),
                            State::Dynamic => {
                                dynamic_states.push(vk::DynamicState::STENCIL_COMPARE_MASK);
                            }
                        }
                        match test.write_mask {
                            State::Static(mask) => builder = builder.write_mask(mask),
                            State::Dynamic => {
                                dynamic_states.push(vk::DynamicState::STENCIL_WRITE_MASK);
                            }
                        }
                        match test.reference {
                            State::Static(value) => builder = builder.reference(value),
                            State::Dynamic => {
                                dynamic_states.push(vk::DynamicState::STENCIL_REFERENCE);
                            }
                        }

                        builder
                    }

                    depth_stencil_state = depth_stencil_state
                        .stencil_test_enable(true)
                        .front(make_stencil_test(&stencil_tests.front, &mut dynamic_states))
                        .back(make_stencil_test(&stencil_tests.back, &mut dynamic_states));
                }

                // Fragment shader stage
                if let Some(shader) = &rasterizer.fragment_shader {
                    fragment_shader_entry =
                        vk::StringArray::<64>::from_bytes(shader.entry().as_bytes());

                    shader_stages.push(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .stage(vk::ShaderStageFlags::FRAGMENT)
                            .module(shader.module().handle())
                            .name(fragment_shader_entry.as_bytes()),
                    );
                }

                // Color blend state
                fn make_blend_attachment(
                    blending: &Option<Blending>,
                    mask: ComponentMask,
                ) -> vk::PipelineColorBlendAttachmentStateBuilder {
                    let builder = vk::PipelineColorBlendAttachmentState::builder();
                    match blending {
                        Some(blending) => builder
                            .blend_enable(true)
                            .src_color_blend_factor(blending.color_src_factor.to_vk())
                            .dst_color_blend_factor(blending.color_dst_factor.to_vk())
                            .color_blend_op(blending.color_op.to_vk())
                            .src_alpha_blend_factor(blending.alpha_src_factor.to_vk())
                            .dst_alpha_blend_factor(blending.alpha_dst_factor.to_vk())
                            .alpha_blend_op(blending.alpha_op.to_vk()),
                        None => builder.blend_enable(false),
                    }
                    .color_write_mask(mask.to_vk())
                }

                match &rasterizer.color_blend {
                    ColorBlend::Logic { op } => {
                        color_blend_state = color_blend_state
                            .logic_op_enable(true)
                            .logic_op((*op).to_vk())
                    }
                    ColorBlend::Blending {
                        blending,
                        write_mask,
                        constants,
                    } => {
                        attachments = (0..color_count)
                            .map(|_| make_blend_attachment(blending, *write_mask))
                            .collect::<Vec<_>>();
                        color_blend_state = color_blend_state.attachments(&attachments);

                        match constants {
                            State::Static(value) => {
                                color_blend_state = color_blend_state.blend_constants(*value)
                            }
                            State::Dynamic => {
                                dynamic_states.push(vk::DynamicState::BLEND_CONSTANTS);
                            }
                        }
                    }
                    ColorBlend::IndependentBlending {
                        blending,
                        constants,
                    } => {
                        assert!(
                            blending.len() == color_count,
                            "independent blending array must have the same length as color attachments"
                        );

                        attachments = blending
                            .iter()
                            .map(|(blending, mask)| make_blend_attachment(blending, *mask))
                            .collect::<Vec<_>>();
                        color_blend_state = color_blend_state.attachments(&attachments);

                        match constants {
                            State::Static(value) => {
                                color_blend_state = color_blend_state.blend_constants(*value)
                            }
                            State::Dynamic => {
                                dynamic_states.push(vk::DynamicState::BLEND_CONSTANTS);
                            }
                        }
                    }
                }

                // Rasterization state
                vk::PipelineRasterizationStateCreateInfo::builder()
                    .rasterizer_discard_enable(false)
                    .depth_clamp_enable(rasterizer.depth_clamp)
                    .polygon_mode(rasterizer.polygin_mode.to_vk())
                    .cull_mode(rasterizer.cull_mode.to_vk())
                    .front_face(rasterizer.front_face.to_vk())
                    .line_width(1.0)
            }
            None => {
                // Rasterization state (discarded)
                vk::PipelineRasterizationStateCreateInfo::builder().rasterizer_discard_enable(true)
            }
        };

        //
        create_info = create_info
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .rasterization_state(&rasterization_state)
            .stages(&shader_stages)
            .layout(descr.layout.handle());

        // Dynamic state
        let pipeline_dynamic_state;
        if !dynamic_states.is_empty() {
            pipeline_dynamic_state =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

            create_info = create_info.dynamic_state(&pipeline_dynamic_state);
        }

        if descr.rasterizer.is_some() {
            create_info = create_info
                .viewport_state(&viewport_state)
                .multisample_state(&multisample_state)
                .depth_stencil_state(&depth_stencil_state)
                .color_blend_state(&color_blend_state);
        }

        let handle = {
            let (mut pipelines, _) = unsafe {
                logical.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&create_info),
                    None,
                )
            }
            .map_err(OutOfDeviceMemory::on_creation)?;

            pipelines.remove(0)
        };

        tracing::debug!(graphics_pipeline = ?handle, "created graphics pipeline");

        Ok(GraphicsPipeline::new(handle, info, self.downgrade()))
    }

    pub fn create_compute_pipeline(
        &self,
        info: ComputePipelineInfo,
    ) -> Result<ComputePipeline, OutOfDeviceMemory> {
        let logical = &self.inner.logical;

        let handle = {
            let name = vk::StringArray::<64>::from_bytes(info.shader.entry().as_bytes());

            let stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(info.shader.module().handle())
                .name(name.as_bytes());

            let info = vk::ComputePipelineCreateInfo::builder()
                .stage(stage)
                .layout(info.layout.handle());

            let (mut pipelines, _) = unsafe {
                logical.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&info),
                    None,
                )
            }
            .map_err(OutOfDeviceMemory::on_creation)?;

            pipelines.remove(0)
        };

        tracing::debug!(compute_pipeline = ?handle, "created compute pipeline");

        Ok(ComputePipeline::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_pipeline(&self, handle: vk::Pipeline) {
        self.logical().destroy_pipeline(handle, None)
    }
}

impl std::fmt::Debug for Device {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.inner.as_ref(), f)
    }
}

impl PartialEq<Device> for Device {
    fn eq(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl PartialEq<Device> for &Device {
    fn eq(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl PartialEq<WeakDevice> for Device {
    fn eq(&self, other: &WeakDevice) -> bool {
        std::ptr::eq(&*self.inner, other.0.as_ptr())
    }
}

impl PartialEq<WeakDevice> for &Device {
    fn eq(&self, other: &WeakDevice) -> bool {
        std::ptr::eq(&*self.inner, other.0.as_ptr())
    }
}

struct Inner {
    logical: vulkanalia::Device,
    physical: vk::PhysicalDevice,
    properties: DeviceProperties,
    features: DeviceFeatures,
    allocator: Mutex<GpuAllocator<vk::DeviceMemory>>,
    descriptors: Mutex<DescriptorAlloc>,
    samplers_cache: FastDashMap<SamplerInfo, Sampler>,
    epochs: Epochs,
}

impl Inner {
    fn wait_idle(&self) -> Result<(), DeviceLost> {
        let old_epochs = self.epochs.next_epoch_all_queues();

        let res = unsafe { self.logical.device_wait_idle() };
        if let Some(vk::ErrorCode::OUT_OF_HOST_MEMORY) = res.err() {
            crate::out_of_host_memory();
        }

        for (queue, epoch) in old_epochs {
            self.epochs.close_epoch(queue, epoch);
        }

        match res {
            Ok(()) => Ok(()),
            Err(vk::ErrorCode::DEVICE_LOST) => Err(DeviceLost),
            Err(e) => crate::unexpected_vulkan_error(e),
        }
    }
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("Device")
                .field("logical", &self.logical.handle())
                .field("physical", &self.physical)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.logical.handle(), f)
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self.wait_idle();

        unsafe {
            self.allocator
                .get_mut()
                .unwrap()
                .cleanup(self.logical.as_memory_device());

            self.descriptors.get_mut().unwrap().cleanup(&self.logical);
        }
    }
}

fn map_memory_device_properties(
    propertis: &DeviceProperties,
    features: &DeviceFeatures,
) -> gpu_alloc::DeviceProperties<'static> {
    let memory = &propertis.memory;
    let limits = &propertis.v1_0.limits;

    let mut max_memory_allocation_size = propertis.v1_1.max_memory_allocation_size;
    if max_memory_allocation_size == 0 {
        max_memory_allocation_size = u64::MAX;
    }

    gpu_alloc::DeviceProperties {
        memory_types: memory.memory_types[..memory.memory_type_count as usize]
            .iter()
            .map(|ty| gpu_alloc::MemoryType {
                heap: ty.heap_index,
                props: gpu_alloc_vulkanalia::memory_properties_from(ty.property_flags),
            })
            .collect(),
        memory_heaps: memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .map(|heap| gpu_alloc::MemoryHeap { size: heap.size })
            .collect(),
        max_memory_allocation_count: limits.max_memory_allocation_count,
        max_memory_allocation_size,
        non_coherent_atom_size: limits.non_coherent_atom_size,
        buffer_device_address: features.v1_2.buffer_device_address != 0,
    }
}

/// An error returned when memory mapping fails.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MapError {
    #[error(transparent)]
    OutOfDeviceMemory(#[from] OutOfDeviceMemory),
    #[error("attempt to map memory block with non-host-visible memory type")]
    NonHostVisible,
    #[error("memory map failed for implementation specific reason")]
    MapFailed,
    #[error("memory mapping failed due to block being already mapped")]
    AlreadyMapped,
}

impl From<gpu_alloc::MapError> for MapError {
    fn from(value: gpu_alloc::MapError) -> Self {
        match value {
            gpu_alloc::MapError::OutOfDeviceMemory => Self::OutOfDeviceMemory(OutOfDeviceMemory),
            gpu_alloc::MapError::OutOfHostMemory => crate::out_of_host_memory(),
            gpu_alloc::MapError::NonHostVisible => Self::NonHostVisible,
            gpu_alloc::MapError::MapFailed => Self::MapFailed,
            gpu_alloc::MapError::AlreadyMapped => Self::AlreadyMapped,
        }
    }
}

/// An error returned when a render pass cannot be created.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CreateRenderPassError {
    #[error(transparent)]
    OutOfDeviceMemory(#[from] OutOfDeviceMemory),

    #[error(
        "attachment index {attachment_index} is out of bounds for the color input {color_index} \
        in the subpass {subpass_index}"
    )]
    ColorAttachmentOutOfBounds {
        attachment_index: u32,
        color_index: usize,
        subpass_index: usize,
    },

    #[error(
        "attachment index {attachment_index} is out of bounds for the depth input \
        in the subpass {subpass_index}"
    )]
    DepthAttachmentOutOfBounds {
        attachment_index: u32,
        subpass_index: usize,
    },
}
