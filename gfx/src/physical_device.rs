use shared::{FastHashMap, FastHashSet};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::InstanceV1_1;

use crate::graphics::Graphics;
use crate::queue::{Queue, QueueFamily, QueueId, QueuesQuery};
use crate::types::{DeviceLost, OutOfDeviceMemory};
use crate::util::ToGfx;

/// A feature that can be requested when creating a device.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DeviceFeature {
    /// Adds a buffer device address to the [`Buffer`].
    ///
    /// [`Buffer`]: crate::Buffer
    BufferDeviceAddress,

    /// Adds ability to update a descriptor binding of
    /// type [`DescriptorType::SampledImage`] after
    /// its descriptor set has been bound.
    ///
    /// [`DescriptorType::SampledImage`]: crate::DescriptorType::SampledImage
    DescriptorBindingSampledImageUpdateAfterBind,

    /// Adds ability to update a descriptor binding of
    /// type [`DescriptorType::StorageImage`] after
    /// its descriptor set has been bound.
    ///
    /// [`DescriptorType::StorageImage`]: crate::DescriptorType::StorageImage
    DescriptorBindingStorageImageUpdateAfterBind,

    /// Adds ability to update a descriptor binding of
    /// type [`DescriptorType::UniformTexelBuffer`] after
    /// its descriptor set has been bound.
    ///
    /// [`DescriptorType::UniformTexelBuffer`]: crate::DescriptorType::UniformTexelBuffer
    DescriptorBindingUniformTexelBufferUpdateAfterBind,

    /// Adds ability to update a descriptor binding of
    /// type [`DescriptorType::StorageTexelBuffer`] after
    /// its descriptor set has been bound.
    ///
    /// [`DescriptorType::StorageTexelBuffer`]: crate::DescriptorType::StorageTexelBuffer
    DescriptorBindingStorageTexelBufferUpdateAfterBind,

    /// Adds ability to update a descriptor binding of
    /// type [`DescriptorType::UniformBuffer`] after
    /// its descriptor set has been bound.
    ///
    /// [`DescriptorType::UniformBuffer`]: crate::DescriptorType::UniformBuffer
    DescriptorBindingUniformBufferUpdateAfterBind,

    /// Adds ability to update a descriptor binding of
    /// type [`DescriptorType::StorageBuffer`] after
    /// its descriptor set has been bound.
    ///
    /// [`DescriptorType::StorageBuffer`]: crate::DescriptorType::StorageBuffer
    DescriptorBindingStorageBufferUpdateAfterBind,

    /// Adds ability to use [`DescriptorBindingFlags::PARTIALLY_BOUND`]
    /// for descriptor bindings.
    ///
    /// [`DescriptorBindingFlags::PARTIALLY_BOUND`]: crate::DescriptorBindingFlags::PARTIALLY_BOUND
    DescriptorBindingPartiallyBound,

    /// Adds ability to query the frame presentation timing.
    DisplayTiming,

    /// Adds [`Min`] and [`Max`] reduction modes to the [`SamplerInfo`].
    ///
    /// [`Min`]: crate::ReductionMode::Min
    /// [`Max`]: crate::ReductionMode::Max
    /// [`SamplerInfo`]: crate::SamplerInfo
    SamplerFilterMinMax,

    /// Must be enabled to use the [`Surface`]
    ///
    /// [`Surface`]: crate::Surface
    SurfacePresentation,

    /// This extension enables C-like structure layout for SPIR-V blocks.
    ScalarBlockLayout,
}

/// A wrapper around Vulkan physical device.
#[derive(Debug)]
pub struct PhysicalDevice {
    handle: vk::PhysicalDevice,
    properties: DeviceProperties,
    features: DeviceFeatures,
}

impl PhysicalDevice {
    pub(crate) unsafe fn new(handle: vk::PhysicalDevice) -> Self {
        let (properties, features) = collect_info(handle);
        PhysicalDevice {
            handle,
            properties,
            features,
        }
    }

    /// Returns an associated graphics instance.
    pub fn graphics(&self) -> &'static Graphics {
        // `PhysicalDevice` can only be created from `Graphics` instance
        unsafe { Graphics::get_unchecked() }
    }

    /// Returns all physical device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Returns all physical device features.
    pub fn features(&self) -> &DeviceFeatures {
        &self.features
    }

    /// Creates a logical device and a set of queues.
    pub fn create_device<Q>(
        self,
        features: &[DeviceFeature],
        queues: Q,
    ) -> Result<(crate::device::Device, Q::Queues), CreateDeviceError<Q::Error>>
    where
        Q: QueuesQuery,
    {
        let graphics = self.graphics();

        let (queue_families, queues_query_state) = queues
            .query(&self.properties.queue_families)
            .map_err(CreateDeviceError::QueueQueryFailed)?;
        let queue_families = queue_families.as_ref();

        let mut device_create_info = vk::DeviceCreateInfo::builder();

        // Collect queries
        let mut priorities = FastHashMap::<usize, Vec<f32>>::default();
        for &(family_idx, count) in queue_families {
            let Some(family) = self.properties.queue_families.get(family_idx) else {
                return Err(CreateDeviceError::UnknownFamilyIndex);
            };
            let priorities = priorities.entry(family_idx).or_default();
            let queue_count = priorities.len() + count;
            if queue_count > family.queue_count as usize {
                return Err(CreateDeviceError::TooManyQueues);
            }

            priorities.resize(queue_count, 1.0f32);
        }

        let queue_create_infos = priorities
            .iter()
            .map(|(&family_idx, priorities)| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(family_idx as u32)
                    .queue_priorities(priorities)
            })
            .collect::<Vec<_>>();

        device_create_info = device_create_info.queue_create_infos(&queue_create_infos);

        // Collect requested features
        let mut requested_features = features.iter().copied().collect::<FastHashSet<_>>();

        let mut extensions = Vec::new();
        let mut require_ext = {
            let supported_extensions = &self.properties.extensions;
            |ext: &vk::Extension| -> bool {
                let ext = &ext.name;
                let supported = supported_extensions.contains(ext);
                if supported && !extensions.contains(&ext.as_ptr()) {
                    extensions.push(ext.as_ptr());
                }
                supported
            }
        };

        let features_v1_0 = vk::PhysicalDeviceFeatures::builder();
        let features_v1_1 = vk::PhysicalDeviceVulkan11Features::builder();
        let mut features_v1_2 = vk::PhysicalDeviceVulkan12Features::builder();
        let features_v1_3 = vk::PhysicalDeviceVulkan13Features::builder();
        let mut features_sbl = vk::PhysicalDeviceScalarBlockLayoutFeatures::builder();
        let mut features_bda = vk::PhysicalDeviceBufferAddressFeaturesEXT::builder();
        let mut features_di = vk::PhysicalDeviceDescriptorIndexingFeatures::builder();

        let mut include_features_v1_2 = false;
        let mut include_features_sbl = false;
        let mut include_features_bda = false;
        let mut include_features_sfmm = false;
        let mut include_features_di = false;

        // === Begin fill feature requirements ===
        if requested_features.remove(&DeviceFeature::BufferDeviceAddress) {
            assert!(
                self.features.v1_2.buffer_device_address != 0,
                "`BufferDeviceAddress` feature is requested but not supported"
            );
            features_v1_2.buffer_device_address = 1;
            include_features_v1_2 = true;
            features_bda.buffer_device_address = 1;
            include_features_bda = true;
        }

        if requested_features.remove(&DeviceFeature::DisplayTiming) {
            let supported = require_ext(&vk::GOOGLE_DISPLAY_TIMING_EXTENSION);
            assert!(
                supported,
                "`DisplayTiming` feature is requested but not supported"
            );
        }

        if requested_features.remove(&DeviceFeature::DescriptorBindingSampledImageUpdateAfterBind) {
            assert!(
                self.features.v1_2.descriptor_binding_sampled_image_update_after_bind != 0,
                "`DescriptorBindingSampledImageUpdateAfterBind` feature is requested but not supported"
            );
            features_v1_2.descriptor_binding_sampled_image_update_after_bind = 1;
            include_features_v1_2 = true;
            features_di.descriptor_binding_sampled_image_update_after_bind = 1;
            include_features_di = true;
        }

        if requested_features.remove(&DeviceFeature::DescriptorBindingStorageImageUpdateAfterBind) {
            assert!(
                self.features.v1_2.descriptor_binding_storage_image_update_after_bind != 0,
                "`DescriptorBindingStorageImageUpdateAfterBind` feature is requested but not supported"
            );
            features_v1_2.descriptor_binding_storage_image_update_after_bind = 1;
            include_features_v1_2 = true;
            features_di.descriptor_binding_storage_image_update_after_bind = 1;
            include_features_di = true;
        }

        if requested_features
            .remove(&DeviceFeature::DescriptorBindingUniformTexelBufferUpdateAfterBind)
        {
            assert!(
                self.features.v1_2.descriptor_binding_uniform_texel_buffer_update_after_bind != 0,
                "`DescriptorBindingUniformTexelBufferUpdateAfterBind` feature is requested but not supported"
            );
            features_v1_2.descriptor_binding_uniform_texel_buffer_update_after_bind = 1;
            include_features_v1_2 = true;
            features_di.descriptor_binding_uniform_texel_buffer_update_after_bind = 1;
            include_features_di = true;
        }

        if requested_features
            .remove(&DeviceFeature::DescriptorBindingStorageTexelBufferUpdateAfterBind)
        {
            assert!(
                self.features.v1_2.descriptor_binding_storage_texel_buffer_update_after_bind != 0,
                "`DescriptorBindingStorageTexelBufferUpdateAfterBind` feature is requested but not supported"
            );
            features_v1_2.descriptor_binding_storage_texel_buffer_update_after_bind = 1;
            include_features_v1_2 = true;
            features_di.descriptor_binding_storage_texel_buffer_update_after_bind = 1;
            include_features_di = true;
        }

        if requested_features.remove(&DeviceFeature::DescriptorBindingUniformBufferUpdateAfterBind)
        {
            assert!(
                self.features.v1_2.descriptor_binding_uniform_buffer_update_after_bind != 0,
                "`DescriptorBindingUniformBufferUpdateAfterBind` feature is requested but not supported"
            );
            features_v1_2.descriptor_binding_uniform_buffer_update_after_bind = 1;
            include_features_v1_2 = true;
            features_di.descriptor_binding_uniform_buffer_update_after_bind = 1;
            include_features_di = true;
        }

        if requested_features.remove(&DeviceFeature::DescriptorBindingStorageBufferUpdateAfterBind)
        {
            assert!(
                self.features.v1_2.descriptor_binding_storage_buffer_update_after_bind != 0,
                "`DescriptorBindingStorageBufferUpdateAfterBind` feature is requested but not supported"
            );
            features_v1_2.descriptor_binding_storage_buffer_update_after_bind = 1;
            include_features_v1_2 = true;
            features_di.descriptor_binding_storage_buffer_update_after_bind = 1;
            include_features_di = true;
        }

        if requested_features.remove(&DeviceFeature::DescriptorBindingPartiallyBound) {
            assert!(
                self.features.v1_2.descriptor_binding_partially_bound != 0,
                "`DescriptorBindingPartiallyBound` feature is requested but not supported"
            );
            features_v1_2.descriptor_binding_partially_bound = 1;
            include_features_v1_2 = true;
            features_di.descriptor_binding_partially_bound = 1;
            include_features_di = true;
        }

        if requested_features.remove(&DeviceFeature::SurfacePresentation) {
            let supported = require_ext(&vk::KHR_SWAPCHAIN_EXTENSION);
            assert!(
                supported,
                "`SurfacePresentation` feature is requested but not supported"
            );
        }

        if requested_features.remove(&DeviceFeature::SamplerFilterMinMax) {
            assert!(
                self.features.v1_2.sampler_filter_minmax != 0,
                "`SamplerFilterMinMax` feature is requested but not supported"
            );
            features_v1_2.sampler_filter_minmax = 1;
            include_features_v1_2 = true;
            include_features_sfmm = true;
        }

        if requested_features.remove(&DeviceFeature::ScalarBlockLayout) {
            assert!(
                self.features.v1_2.scalar_block_layout != 0,
                "`ScalarBlockLayout` feature is requested but not supported"
            );
            features_v1_2.scalar_block_layout = 1;
            include_features_v1_2 = true;
            features_sbl.scalar_block_layout = 1;
            include_features_sbl = true;
        }
        // === End fill feature requirements ===

        device_create_info = device_create_info.enabled_features(&features_v1_0);

        // Prevent requiring duplicate blocks
        if graphics.vk1_2() {
            include_features_sbl = false;
            include_features_bda = false;
            include_features_sfmm = false;
            include_features_di = false;
        } else {
            include_features_v1_2 = false;
        }

        // Require requested blocks
        if include_features_v1_2 {
            device_create_info = device_create_info.push_next(&mut features_v1_2);
        }
        if include_features_sbl {
            let supported = require_ext(&vk::EXT_SCALAR_BLOCK_LAYOUT_EXTENSION);
            assert!(
                supported,
                "`ScalarBlockLayout` feature is requested but not supported"
            );
            device_create_info = device_create_info.push_next(&mut features_sbl);
        }
        if include_features_bda {
            let supported = require_ext(&vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION);
            assert!(
                supported,
                "`BufferDeviceAddress` feature is requested but not supported"
            );
            device_create_info = device_create_info.push_next(&mut features_bda);
        }
        if include_features_sfmm {
            let supported = require_ext(&vk::EXT_SAMPLER_FILTER_MINMAX_EXTENSION);
            assert!(
                supported,
                "`SamplerFilterMinMax` feature is requested but not supported"
            )
        }
        if include_features_di {
            let supported = require_ext(&vk::EXT_DESCRIPTOR_INDEXING_EXTENSION);
            assert!(
                supported,
                "Descriptor indexing feature is requested but not supported"
            );
            device_create_info = device_create_info.push_next(&mut features_di);
        }

        // Ensure all required features are supported
        assert!(
            requested_features.is_empty(),
            "some features are required but not supported: {}",
            requested_features
                .into_iter()
                .map(|f| format!("{f:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );

        device_create_info = device_create_info.enabled_extension_names(&extensions);

        // Create device
        let logical = unsafe {
            graphics
                .instance()
                .create_device(self.handle, &device_create_info, None)
                .map_err(|e| match e {
                    vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
                    vk::ErrorCode::OUT_OF_DEVICE_MEMORY => {
                        CreateDeviceError::from(OutOfDeviceMemory)
                    }
                    vk::ErrorCode::INITIALIZATION_FAILED => CreateDeviceError::InitializationFailed,
                    vk::ErrorCode::DEVICE_LOST => CreateDeviceError::from(DeviceLost),
                    _ => crate::unexpected_vulkan_error(e),
                })?
        };
        let device = crate::device::Device::new(
            logical,
            self.handle,
            self.properties,
            DeviceFeatures {
                v1_0: features_v1_0.build(),
                v1_1: features_v1_1.build(),
                v1_2: features_v1_2.build(),
                v1_3: features_v1_3.build(),
            },
            queue_families.iter().flat_map(|&(family, queue_count)| {
                let family = family as u32;
                (0..queue_count).map(move |index| {
                    let index = index as u32;
                    QueueId { family, index }
                })
            }),
        );

        tracing::debug!(?device, "created device");

        // Create queues
        let queue_families = queue_families
            .iter()
            .map(|&(family_idx, count)| {
                let capabilities = device.properties().queue_families[family_idx]
                    .queue_flags
                    .to_gfx();

                QueueFamily {
                    capabilities,
                    queues: (0..count)
                        .map(|index| {
                            let family_idx = family_idx as u32;
                            let queue_idx = index as u32;
                            let handle =
                                unsafe { device.logical().get_device_queue(family_idx, queue_idx) };

                            Queue::new(handle, family_idx, queue_idx, capabilities, device.clone())
                        })
                        .collect(),
                }
            })
            .collect();

        let queues = Q::collect(queues_query_state, queue_families);

        Ok((device, queues))
    }
}

/// All physical device properties.
#[derive(Debug)]
pub struct DeviceProperties {
    pub extensions: FastHashSet<vk::ExtensionName>,
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub v1_0: vk::PhysicalDeviceProperties,
    pub v1_1: vk::PhysicalDeviceVulkan11Properties,
    pub v1_2: vk::PhysicalDeviceVulkan12Properties,
    pub v1_3: vk::PhysicalDeviceVulkan13Properties,
}

unsafe impl Sync for DeviceProperties {}
unsafe impl Send for DeviceProperties {}

/// All physical device features.
#[derive(Debug)]
pub struct DeviceFeatures {
    pub v1_0: vk::PhysicalDeviceFeatures,
    pub v1_1: vk::PhysicalDeviceVulkan11Features,
    pub v1_2: vk::PhysicalDeviceVulkan12Features,
    pub v1_3: vk::PhysicalDeviceVulkan13Features,
}

unsafe impl Sync for DeviceFeatures {}
unsafe impl Send for DeviceFeatures {}

unsafe fn collect_info(handle: vk::PhysicalDevice) -> (DeviceProperties, DeviceFeatures) {
    let graphics = Graphics::get_unchecked();
    let instance = graphics.instance();
    let (vk1_1, vk1_2, vk1_3) = (graphics.vk1_1(), graphics.vk1_2(), graphics.vk1_3());

    let extensions = instance
        .enumerate_device_extension_properties(handle, None)
        .unwrap()
        .into_iter()
        .map(|item| item.extension_name)
        .collect::<FastHashSet<_>>();
    let has_device_ext = |ext: &vk::Extension| -> bool { extensions.contains(&ext.name) };

    let properties_v1_0;
    let mut properties_v1_1 = vk::PhysicalDeviceVulkan11Properties::builder();
    let mut properties_v1_2 = vk::PhysicalDeviceVulkan12Properties::builder();
    let mut properties_v1_3 = vk::PhysicalDeviceVulkan13Properties::builder();
    let mut properties_di = vk::PhysicalDeviceDescriptorIndexingProperties::builder();
    let mut properties_mt3 = vk::PhysicalDeviceMaintenance3Properties::builder();

    let features_v1_0;
    let mut features_v1_1 = vk::PhysicalDeviceVulkan11Features::builder();
    let mut features_v1_2 = vk::PhysicalDeviceVulkan12Features::builder();
    let mut features_v1_3 = vk::PhysicalDeviceVulkan13Features::builder();
    let mut features_di = vk::PhysicalDeviceDescriptorIndexingFeatures::builder();
    let mut features_sbl = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::builder();
    let mut features_bda = vk::PhysicalDeviceBufferAddressFeaturesEXT::builder();

    // Query info
    if vk1_1
        || instance
            .extensions()
            .contains(&vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name)
    {
        {
            let mut properties2 = vk::PhysicalDeviceProperties2::builder();
            let mut features2 = vk::PhysicalDeviceFeatures2::builder();

            // Core properties and features
            if vk1_1 {
                properties2 = properties2.push_next(&mut properties_v1_1);
                features2 = features2.push_next(&mut features_v1_1);
            }
            if vk1_2 {
                properties2 = properties2.push_next(&mut properties_v1_2);
                features2 = features2.push_next(&mut features_v1_2);
            }
            if vk1_3 {
                properties2 = properties2.push_next(&mut properties_v1_3);
                features2 = features2.push_next(&mut features_v1_3);
            }

            // Extension properties and features
            if !vk1_1 && has_device_ext(&vk::KHR_MAINTENANCE3_EXTENSION) {
                properties2 = properties2.push_next(&mut properties_mt3);
            }
            if !vk1_2 && has_device_ext(&vk::EXT_DESCRIPTOR_INDEXING_EXTENSION) {
                properties2 = properties2.push_next(&mut properties_di);
                features2 = features2.push_next(&mut features_di);
            }
            if !vk1_2 && has_device_ext(&vk::EXT_SCALAR_BLOCK_LAYOUT_EXTENSION) {
                features2 = features2.push_next(&mut features_sbl);
            }
            if !vk1_2 && has_device_ext(&vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION) {
                features2 = features2.push_next(&mut features_bda);
            }

            // Query extended info
            instance.get_physical_device_properties2(handle, &mut properties2);
            instance.get_physical_device_features2(handle, &mut features2);

            properties_v1_0 = properties2.properties;
            features_v1_0 = features2.features;
        }

        properties_di.next = std::ptr::null_mut();
        properties_mt3.next = std::ptr::null_mut();
        properties_v1_3.next = std::ptr::null_mut();
        properties_v1_2.next = std::ptr::null_mut();
        properties_v1_1.next = std::ptr::null_mut();

        features_bda.next = std::ptr::null_mut();
        features_sbl.next = std::ptr::null_mut();
        features_di.next = std::ptr::null_mut();
        features_v1_3.next = std::ptr::null_mut();
        features_v1_2.next = std::ptr::null_mut();
        features_v1_1.next = std::ptr::null_mut();
    } else {
        // Query basic info
        properties_v1_0 = instance.get_physical_device_properties(handle);
        features_v1_0 = instance.get_physical_device_features(handle);
    }

    // Other info
    let queue_families = instance.get_physical_device_queue_family_properties(handle);
    let memory_properties = instance.get_physical_device_memory_properties(handle);

    // Map extensions to core
    if !vk1_1 && has_device_ext(&vk::KHR_MAINTENANCE3_EXTENSION) {
        properties_v1_1.max_per_set_descriptors = properties_mt3.max_per_set_descriptors;
        properties_v1_1.max_memory_allocation_size = properties_mt3.max_memory_allocation_size;
    }
    if !vk1_2 && has_device_ext(&vk::EXT_DESCRIPTOR_INDEXING_EXTENSION) {
        properties_v1_2.max_update_after_bind_descriptors_in_all_pools =
            properties_di.max_update_after_bind_descriptors_in_all_pools;
        properties_v1_2.shader_uniform_buffer_array_non_uniform_indexing_native =
            properties_di.shader_uniform_buffer_array_non_uniform_indexing_native;
        properties_v1_2.shader_sampled_image_array_non_uniform_indexing_native =
            properties_di.shader_sampled_image_array_non_uniform_indexing_native;
        properties_v1_2.shader_storage_buffer_array_non_uniform_indexing_native =
            properties_di.shader_storage_buffer_array_non_uniform_indexing_native;
        properties_v1_2.shader_storage_image_array_non_uniform_indexing_native =
            properties_di.shader_storage_image_array_non_uniform_indexing_native;
        properties_v1_2.shader_input_attachment_array_non_uniform_indexing_native =
            properties_di.shader_input_attachment_array_non_uniform_indexing_native;
        properties_v1_2.robust_buffer_access_update_after_bind =
            properties_di.robust_buffer_access_update_after_bind;
        properties_v1_2.quad_divergent_implicit_lod = properties_di.quad_divergent_implicit_lod;
        properties_v1_2.max_per_stage_descriptor_update_after_bind_samplers =
            properties_di.max_per_stage_descriptor_update_after_bind_samplers;
        properties_v1_2.max_per_stage_descriptor_update_after_bind_uniform_buffers =
            properties_di.max_per_stage_descriptor_update_after_bind_uniform_buffers;
        properties_v1_2.max_per_stage_descriptor_update_after_bind_storage_buffers =
            properties_di.max_per_stage_descriptor_update_after_bind_storage_buffers;
        properties_v1_2.max_per_stage_descriptor_update_after_bind_sampled_images =
            properties_di.max_per_stage_descriptor_update_after_bind_sampled_images;
        properties_v1_2.max_per_stage_descriptor_update_after_bind_storage_images =
            properties_di.max_per_stage_descriptor_update_after_bind_storage_images;
        properties_v1_2.max_per_stage_descriptor_update_after_bind_input_attachments =
            properties_di.max_per_stage_descriptor_update_after_bind_input_attachments;
        properties_v1_2.max_per_stage_update_after_bind_resources =
            properties_di.max_per_stage_update_after_bind_resources;
        properties_v1_2.max_descriptor_set_update_after_bind_samplers =
            properties_di.max_descriptor_set_update_after_bind_samplers;
        properties_v1_2.max_descriptor_set_update_after_bind_uniform_buffers =
            properties_di.max_descriptor_set_update_after_bind_uniform_buffers;
        properties_v1_2.max_descriptor_set_update_after_bind_uniform_buffers_dynamic =
            properties_di.max_descriptor_set_update_after_bind_uniform_buffers_dynamic;
        properties_v1_2.max_descriptor_set_update_after_bind_storage_buffers =
            properties_di.max_descriptor_set_update_after_bind_storage_buffers;
        properties_v1_2.max_descriptor_set_update_after_bind_storage_buffers_dynamic =
            properties_di.max_descriptor_set_update_after_bind_storage_buffers_dynamic;
        properties_v1_2.max_descriptor_set_update_after_bind_sampled_images =
            properties_di.max_descriptor_set_update_after_bind_sampled_images;
        properties_v1_2.max_descriptor_set_update_after_bind_storage_images =
            properties_di.max_descriptor_set_update_after_bind_storage_images;
        properties_v1_2.max_descriptor_set_update_after_bind_input_attachments =
            properties_di.max_descriptor_set_update_after_bind_input_attachments;

        features_v1_2.descriptor_indexing = 1;
        features_v1_2.shader_input_attachment_array_dynamic_indexing =
            features_di.shader_input_attachment_array_dynamic_indexing;
        features_v1_2.shader_uniform_texel_buffer_array_dynamic_indexing =
            features_di.shader_uniform_texel_buffer_array_dynamic_indexing;
        features_v1_2.shader_storage_texel_buffer_array_dynamic_indexing =
            features_di.shader_storage_texel_buffer_array_dynamic_indexing;
        features_v1_2.shader_uniform_buffer_array_non_uniform_indexing =
            features_di.shader_uniform_buffer_array_non_uniform_indexing;
        features_v1_2.shader_sampled_image_array_non_uniform_indexing =
            features_di.shader_sampled_image_array_non_uniform_indexing;
        features_v1_2.shader_storage_buffer_array_non_uniform_indexing =
            features_di.shader_storage_buffer_array_non_uniform_indexing;
        features_v1_2.shader_storage_image_array_non_uniform_indexing =
            features_di.shader_storage_image_array_non_uniform_indexing;
        features_v1_2.shader_input_attachment_array_non_uniform_indexing =
            features_di.shader_input_attachment_array_non_uniform_indexing;
        features_v1_2.shader_uniform_texel_buffer_array_non_uniform_indexing =
            features_di.shader_uniform_texel_buffer_array_non_uniform_indexing;
        features_v1_2.shader_storage_texel_buffer_array_non_uniform_indexing =
            features_di.shader_storage_texel_buffer_array_non_uniform_indexing;
        features_v1_2.descriptor_binding_uniform_buffer_update_after_bind =
            features_di.descriptor_binding_uniform_buffer_update_after_bind;
        features_v1_2.descriptor_binding_sampled_image_update_after_bind =
            features_di.descriptor_binding_sampled_image_update_after_bind;
        features_v1_2.descriptor_binding_storage_image_update_after_bind =
            features_di.descriptor_binding_storage_image_update_after_bind;
        features_v1_2.descriptor_binding_storage_buffer_update_after_bind =
            features_di.descriptor_binding_storage_buffer_update_after_bind;
        features_v1_2.descriptor_binding_uniform_texel_buffer_update_after_bind =
            features_di.descriptor_binding_uniform_texel_buffer_update_after_bind;
        features_v1_2.descriptor_binding_storage_texel_buffer_update_after_bind =
            features_di.descriptor_binding_storage_texel_buffer_update_after_bind;
        features_v1_2.descriptor_binding_update_unused_while_pending =
            features_di.descriptor_binding_update_unused_while_pending;
        features_v1_2.descriptor_binding_partially_bound =
            features_di.descriptor_binding_partially_bound;
        features_v1_2.descriptor_binding_variable_descriptor_count =
            features_di.descriptor_binding_variable_descriptor_count;
        features_v1_2.runtime_descriptor_array = features_di.runtime_descriptor_array;
    }
    if !vk1_2 && has_device_ext(&vk::EXT_SAMPLER_FILTER_MINMAX_EXTENSION) {
        features_v1_2.sampler_filter_minmax = 1;
    }
    if !vk1_2 && has_device_ext(&vk::EXT_SCALAR_BLOCK_LAYOUT_EXTENSION) {
        features_v1_2.scalar_block_layout = features_sbl.scalar_block_layout;
    }
    if !vk1_2 && has_device_ext(&vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION) {
        features_v1_2.buffer_device_address = features_bda.buffer_device_address;
        features_v1_2.buffer_device_address_capture_replay =
            features_bda.buffer_device_address_capture_replay;
        features_v1_2.buffer_device_address_multi_device =
            features_bda.buffer_device_address_multi_device;
    }

    let properties = DeviceProperties {
        extensions,
        queue_families,
        memory: memory_properties,
        v1_0: properties_v1_0,
        v1_1: properties_v1_1.build(),
        v1_2: properties_v1_2.build(),
        v1_3: properties_v1_3.build(),
    };
    let features = DeviceFeatures {
        v1_0: features_v1_0,
        v1_1: features_v1_1.build(),
        v1_2: features_v1_2.build(),
        v1_3: features_v1_3.build(),
    };
    (properties, features)
}

/// An error returned when a logical device could not be created.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CreateDeviceError<E> {
    #[error(transparent)]
    OutOfDeviceMemory(#[from] OutOfDeviceMemory),
    #[error(transparent)]
    DeviceLost(#[from] DeviceLost),
    #[error("queue query failed")]
    QueueQueryFailed(#[source] E),
    #[error("requested queue family index is out of bounds")]
    UnknownFamilyIndex,
    #[error("requested too many queues in a single family")]
    TooManyQueues,
    #[error("device initialization could not be completed for implementation-specific reasons")]
    InitializationFailed,
}
