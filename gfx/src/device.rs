use std::sync::{Arc, Mutex, Weak};

use anyhow::Result;
use gpu_alloc::{GpuAllocator, MemoryBlock};
use gpu_alloc_vulkanalia::AsMemoryDevice;
use shared::util::WithDefer;
use smallvec::SmallVec;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{DeviceV1_1, DeviceV1_2};
use winit::window::Window;

use crate::graphics::Graphics;
use crate::physical_device::{DeviceFeatures, DeviceProperties};
use crate::resources::{
    Buffer, BufferInfo, ComputePipeline, ComputePipelineInfo, DescriptorSetLayout,
    DescriptorSetLayoutInfo, Fence, FenceState, Framebuffer, FramebufferInfo, Image, ImageInfo,
    ImageView, ImageViewInfo, ImageViewType, MappableBuffer, PipelineLayout, PipelineLayoutInfo,
    RenderPass, RenderPassInfo, Semaphore, ShaderModule, ShaderModuleInfo,
};
use crate::surface::Surface;
use crate::types::DeviceAddress;

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

#[derive(Clone)]
#[repr(transparent)]
pub struct Device {
    inner: Arc<Inner>,
}

impl Device {
    pub fn new(
        logical: vulkanalia::Device,
        physical: vk::PhysicalDevice,
        properties: DeviceProperties,
        features: DeviceFeatures,
    ) -> Self {
        let allocator = Mutex::new(GpuAllocator::new(
            gpu_alloc::Config::i_am_prototyping(),
            map_memory_device_properties(&properties, &features),
        ));

        Self {
            inner: Arc::new(Inner {
                logical,
                physical,
                properties,
                features,
                allocator,
            }),
        }
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

    pub fn properties(&self) -> &DeviceProperties {
        &self.inner.properties
    }

    pub fn features(&self) -> &DeviceFeatures {
        &self.inner.features
    }

    pub fn downgrade(&self) -> WeakDevice {
        WeakDevice(Arc::downgrade(&self.inner))
    }

    pub fn wait_idle(&self) -> Result<()> {
        self.inner.wait_idle()
    }

    pub fn create_semaphore(&self) -> Result<Semaphore> {
        let logical = &self.inner.logical;

        let info = vk::SemaphoreCreateInfo::builder();
        let handle = unsafe { logical.create_semaphore(&info, None) }?;

        tracing::debug!(semaphore = ?handle, "created semaphore");

        Ok(Semaphore::new(handle, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_semaphore(&self, handle: vk::Semaphore) {
        self.inner.logical.destroy_semaphore(handle, None);
    }

    pub fn create_fence(&self) -> Result<Fence> {
        let logical = &self.inner.logical;

        let info = vk::FenceCreateInfo::builder();
        let handle = unsafe { logical.create_fence(&info, None) }?;

        tracing::debug!(fence = ?handle, "created fence");

        Ok(Fence::new(handle, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_fence(&self, handle: vk::Fence) {
        self.inner.logical.destroy_fence(handle, None);
    }

    pub fn update_armed_fence_state(&self, fence: &mut Fence) -> Result<bool> {
        let status = unsafe { self.inner.logical.get_fence_status(fence.handle()) }?;
        match status {
            vk::SuccessCode::SUCCESS => {
                let _epoch = fence.set_signalled()?;
                // TODO: update epoch
                Ok(true)
            }
            vk::SuccessCode::NOT_READY => Ok(false),
            c => panic!("unexpected status code: {c:?}"),
        }
    }

    pub fn reset_fences(&self, fences: &mut [&mut Fence]) -> Result<()> {
        let handles = fences
            .iter_mut()
            .map(|fence| {
                if matches!(fence.state(), FenceState::Armed { .. }) {
                    match self.update_armed_fence_state(fence) {
                        // Signalled -> ok
                        Ok(true) => {}
                        // Armed and not signalled yet -> error
                        Ok(false) => return Err(anyhow::anyhow!("armed fence cannot be reset")),
                        // Failed to check -> error
                        Err(e) => return Err(e),
                    }
                }
                Ok(fence.handle())
            })
            .collect::<Result<SmallVec<[_; 16]>>>()?;

        unsafe { self.inner.logical.reset_fences(&handles) }?;

        for fence in fences {
            fence.set_unsignalled()?;
        }

        Ok(())
    }

    pub fn wait_fences(&self, fences: &mut [&mut Fence], wait_all: bool) -> Result<()> {
        let handles = fences
            .iter()
            .filter_map(|fence| match fence.state() {
                // Waiting for an unarmed fence -> error (preventing deadlock)
                FenceState::Unsignalled => {
                    Some(Err(anyhow::anyhow!("waiting for an unarmed fence")))
                }
                // Waiting for an armed fence -> ok
                FenceState::Armed { .. } => Some(Ok(fence.handle())),
                // Already signalled fences could be skipped
                FenceState::Signalled => None,
            })
            .collect::<Result<SmallVec<[_; 16]>>>()?;

        if handles.is_empty() {
            return Ok(());
        }

        unsafe {
            self.inner
                .logical
                .wait_for_fences(&handles, wait_all, u64::MAX)
        }?;

        let all_signalled = wait_all || handles.len() == 1;
        for fence in fences {
            if all_signalled || self.update_armed_fence_state(fence)? {
                fence.set_signalled()?;
            }
        }

        // TODO: update epochs

        Ok(())
    }

    pub fn create_surface(&self, window: Arc<Window>) -> Result<Surface> {
        let handle = self.graphics().create_raw_surface(&window)?;

        tracing::debug!(surface = ?handle, "created surface");

        Surface::new(handle, window, self)
    }

    pub fn create_buffer(&self, info: BufferInfo) -> Result<Buffer> {
        self.create_buffer_impl(info, None)
            .map(MappableBuffer::freeze)
    }

    pub fn create_mappable_buffer(
        &self,
        info: BufferInfo,
        memory_usage: gpu_alloc::UsageFlags,
    ) -> Result<MappableBuffer> {
        self.create_buffer_impl(info, Some(memory_usage))
    }

    fn create_buffer_impl(
        &self,
        info: BufferInfo,
        memory_usage: Option<gpu_alloc::UsageFlags>,
    ) -> Result<MappableBuffer> {
        let logical = &self.inner.logical;

        let mut memory_usage = memory_usage.unwrap_or_else(gpu_alloc::UsageFlags::empty);
        let has_device_address = info
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        if has_device_address {
            anyhow::ensure!(
                self.inner.features.v1_2.buffer_device_address != 0,
                "`SHADER_DEVICE_ADDRESS` buffer usage requires `BufferDeviceAddress`
                feature"
            );
            memory_usage |= gpu_alloc::UsageFlags::DEVICE_ADDRESS;
        }

        let handle = {
            let info = vk::BufferCreateInfo::builder()
                .size(info.size)
                .usage(info.usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            unsafe { logical.create_buffer(&info, None)? }
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
                usage: memory_usage,
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
        }?;

        unsafe { logical.bind_buffer_memory(*handle, *block.memory(), block.offset())? };

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
            memory_usage,
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
            .dealloc(self.inner.logical.as_memory_device(), block);

        self.inner.logical.destroy_buffer(handle, None);
    }

    pub fn create_image(&self, info: ImageInfo) -> Result<Image> {
        let logical = &self.inner.logical;

        let handle = {
            let info = vk::ImageCreateInfo::builder()
                .image_type(info.extent.into())
                .format(info.format.into())
                .extent(vk::Extent3D::from(info.extent))
                .mip_levels(info.mip_levels)
                .samples(info.samples.into())
                .array_layers(info.array_layers)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(info.usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            unsafe { logical.create_image(&info, None) }?
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
        }?;

        unsafe { logical.bind_image_memory(*handle, *block.memory(), block.offset())? };

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
            .dealloc(self.inner.logical.as_memory_device(), block);

        self.inner.logical.destroy_image(handle, None)
    }

    pub fn create_image_view(&self, info: ImageViewInfo) -> Result<ImageView> {
        let logical = &self.inner.logical;

        let handle = {
            let info = vk::ImageViewCreateInfo::builder()
                .image(info.image.handle())
                .format(info.image.info().format.into())
                .view_type(info.ty.into())
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(info.range.aspect_mask)
                        .base_mip_level(info.range.base_mip_level)
                        .level_count(info.range.level_count)
                        .base_array_layer(info.range.base_array_layer)
                        .layer_count(info.range.layer_count),
                )
                .components(vk::ComponentMapping::from(info.mapping));

            unsafe { logical.create_image_view(&info, None) }?
        };

        tracing::debug!(image_view = ?handle, "created image view");

        Ok(ImageView::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_image_view(&self, handle: vk::ImageView) {
        self.inner.logical.destroy_image_view(handle, None);
    }

    pub fn create_shader_module(&self, info: ShaderModuleInfo) -> Result<ShaderModule> {
        let handle = {
            let info = vk::ShaderModuleCreateInfo::builder()
                .code_size(info.data.len() * 4)
                .code(&info.data);

            unsafe { self.logical().create_shader_module(&info, None) }?
        };

        tracing::debug!(shader_module = ?handle, "created shader module");

        Ok(ShaderModule::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_shader_module(&self, handle: vk::ShaderModule) {
        self.inner.logical.destroy_shader_module(handle, None);
    }

    pub fn create_render_pass(&self, info: RenderPassInfo) -> Result<RenderPass> {
        let mut subpass_attachments = Vec::new();

        let mut subpasses = SmallVec::<[_; 4]>::with_capacity(info.subpasses.len());
        for (subpass_index, subpass) in info.subpasses.iter().enumerate() {
            let color_offset = subpass_attachments.len();
            subpass_attachments.reserve(subpass.colors.len() + subpass.depth.is_some() as usize);

            for (color_index, &(i, layout)) in subpass.colors.iter().enumerate() {
                anyhow::ensure!(
                    (i as usize) < info.attachments.len(),
                    "attachment index {i} is out of bounds for the color input {color_index} \
                    in the subpass {subpass_index}"
                );
                subpass_attachments.push(
                    vk::AttachmentReference::builder()
                        .attachment(i)
                        .layout(layout.into()),
                );
            }

            let depths_offset = subpass_attachments.len();
            if let Some((i, layout)) = subpass.depth {
                anyhow::ensure!(
                    (i as usize) < info.attachments.len(),
                    "attachment index {i} is out of bounds for the depths input \
                    in the subpass {subpass_index}"
                );
                subpass_attachments.push(
                    vk::AttachmentReference::builder()
                        .attachment(i)
                        .layout(layout.into()),
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
                    .format(info.format.into())
                    .load_op(info.load_op.into())
                    .store_op(info.store_op.into())
                    .initial_layout(
                        info.initial_layout
                            .map(Into::into)
                            .unwrap_or(vk::ImageLayout::UNDEFINED),
                    )
                    .final_layout(info.final_layout.into())
                    .samples(vk::SampleCountFlags::_1)
            })
            .collect::<Vec<_>>();

        let dependencies = info
            .dependencies
            .iter()
            .map(|info| {
                vk::SubpassDependency::builder()
                    .src_subpass(info.src.unwrap_or(vk::SUBPASS_EXTERNAL))
                    .dst_subpass(info.dst.unwrap_or(vk::SUBPASS_EXTERNAL))
                    .src_stage_mask(info.src_stages)
                    .dst_stage_mask(info.dst_stages)
                    .build()
            })
            .collect::<Vec<_>>();

        let handle = {
            let info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies);

            unsafe { self.logical().create_render_pass(&info, None) }?
        };

        tracing::debug!(render_pass = ?handle, "created render pass");

        Ok(RenderPass::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_render_pass(&self, handle: vk::RenderPass) {
        self.logical().destroy_render_pass(handle, None);
    }

    pub fn create_framebuffer(&self, info: FramebufferInfo) -> Result<Framebuffer> {
        anyhow::ensure!(
            info.attachments
                .iter()
                .all(|view| view.info().ty == ImageViewType::D2),
            "all image views must be 2d images"
        );

        anyhow::ensure!(
            info.attachments.iter().all(|view| {
                let extent: vk::Extent2D = view.info().image.info().extent.into();
                extent.width >= info.extent.width && extent.height >= info.extent.height
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
                .width(info.extent.width)
                .height(info.extent.height)
                .layers(1);

            unsafe { self.logical().create_framebuffer(&info, None) }?
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
    ) -> Result<DescriptorSetLayout> {
        let logical = &self.inner.logical;

        let handle = {
            let flags;
            let mut flags_info;

            let mut create_info = vk::DescriptorSetLayoutCreateInfo::builder().flags(info.flags);

            if self.graphics().vk1_2() {
                if info.bindings.iter().any(|b| {
                    b.flags
                        .contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND)
                }) {
                    anyhow::ensure!(
                        info.flags
                            .contains(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL),
                        "`UPDATE_AFTER_BIND_POOL` flag must be set in descriptor set layout \
                        create info flags"
                    );
                }

                flags = info
                    .bindings
                    .iter()
                    .map(|b| b.flags)
                    .collect::<SmallVec<[_; 8]>>();

                flags_info =
                    vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);
                create_info = create_info.push_next(&mut flags_info);
            } else {
                anyhow::ensure!(
                    info.bindings.iter().all(|b| b.flags.is_empty()),
                    "Vulkan 1.2 is required for non-empty `DescriptorBindingFlags`"
                );
            }

            let bindings = info
                .bindings
                .iter()
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(binding.binding)
                        .descriptor_count(binding.count)
                        .descriptor_type(binding.ty.into())
                        .stage_flags(binding.stages)
                })
                .collect::<SmallVec<[_; 8]>>();

            create_info = create_info.bindings(&bindings);

            unsafe { logical.create_descriptor_set_layout(&create_info, None) }?
        };

        tracing::debug!(descriptor_set_layout = ?handle, "created descriptor set layout");

        Ok(DescriptorSetLayout::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_descriptor_set_layout(&self, handle: vk::DescriptorSetLayout) {
        self.logical().destroy_descriptor_set_layout(handle, None)
    }

    pub fn create_pipeline_layout(&self, info: PipelineLayoutInfo) -> Result<PipelineLayout> {
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
                .map(|c| {
                    vk::PushConstantRange::builder()
                        .stage_flags(c.stages)
                        .offset(c.offset)
                        .size(c.size)
                })
                .collect::<SmallVec<[_; 8]>>();

            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&sets)
                .push_constant_ranges(&push_constants);

            unsafe { logical.create_pipeline_layout(&info, None) }?
        };

        tracing::debug!(pipeline_layout = ?handle, "created pipeline layout");

        Ok(PipelineLayout::new(handle, info, self.downgrade()))
    }

    pub(crate) unsafe fn destroy_pipeline_layout(&self, handle: vk::PipelineLayout) {
        self.logical().destroy_pipeline_layout(handle, None)
    }

    pub fn create_compute_pipeline(&self, info: ComputePipelineInfo) -> Result<ComputePipeline> {
        let logical = &self.inner.logical;

        let handle = {
            let name = vk::StringArray::<128>::from_bytes(info.shader.entry().as_bytes());

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
            }?;

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
}

impl Inner {
    fn wait_idle(&self) -> Result<()> {
        // TODO: wait queues
        unsafe { self.logical.device_wait_idle()? };
        // TODO: reset queues?
        Ok(())
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

            // TODO: destroy device?
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
