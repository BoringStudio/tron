use std::collections::HashSet;

use vulkanalia::prelude::v1_0::*;

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

pub type AllExtensions = (
    BufferDeviceAddressExtension,
    DescriptorIndexingExtension,
    DisplayTimingExtension,
    SamplerFilterMinMaxExtension,
    ScalarBlockLayoutExtension,
    SurfacePresentationExtension,
);

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

pub struct BufferDeviceAddressExtension;

impl VulkanExtension for BufferDeviceAddressExtension {
    const META: &'static vk::Extension = &vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION;

    type Core = VulkanCore<1, 2>;
    type ExtensionFeatures = WithFeatures<vk::PhysicalDeviceBufferDeviceAddressFeatures>;
    type ExtensionProperties = NoProperties;

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut DeviceFeatures,
    ) {
        let core_features = &mut core_features.v1_2;

        core_features.buffer_device_address = extension_features.buffer_device_address;
        core_features.buffer_device_address_capture_replay =
            extension_features.buffer_device_address_capture_replay;
        core_features.buffer_device_address_multi_device =
            extension_features.buffer_device_address_multi_device;
    }

    fn process_features<L, I>(
        availabe: &DeviceFeatures,
        enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        let enabled = enabled.get_mut();
        let mut changed = false;
        if required.remove(&DeviceFeature::BufferDeviceAddress) {
            assert!(
                availabe.v1_2.buffer_device_address != 0,
                "`BufferDeviceAddress` feature is required but not supported"
            );
            enabled.buffer_device_address = 1;
            changed = true;
        }
        changed
    }
}

pub struct DescriptorIndexingExtension;

impl VulkanExtension for DescriptorIndexingExtension {
    const META: &'static vk::Extension = &vk::EXT_DESCRIPTOR_INDEXING_EXTENSION;

    type Core = VulkanCore<1, 2>;
    type ExtensionFeatures = WithFeatures<vk::PhysicalDeviceDescriptorIndexingFeatures>;
    type ExtensionProperties = WithProperties<vk::PhysicalDeviceDescriptorIndexingProperties>;

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut DeviceFeatures,
    ) {
        let core_features = &mut core_features.v1_2;

        core_features.descriptor_indexing = 1;
        core_features.shader_input_attachment_array_dynamic_indexing =
            extension_features.shader_input_attachment_array_dynamic_indexing;
        core_features.shader_uniform_texel_buffer_array_dynamic_indexing =
            extension_features.shader_uniform_texel_buffer_array_dynamic_indexing;
        core_features.shader_storage_texel_buffer_array_dynamic_indexing =
            extension_features.shader_storage_texel_buffer_array_dynamic_indexing;
        core_features.shader_uniform_buffer_array_non_uniform_indexing =
            extension_features.shader_uniform_buffer_array_non_uniform_indexing;
        core_features.shader_sampled_image_array_non_uniform_indexing =
            extension_features.shader_sampled_image_array_non_uniform_indexing;
        core_features.shader_storage_buffer_array_non_uniform_indexing =
            extension_features.shader_storage_buffer_array_non_uniform_indexing;
        core_features.shader_storage_image_array_non_uniform_indexing =
            extension_features.shader_storage_image_array_non_uniform_indexing;
        core_features.shader_input_attachment_array_non_uniform_indexing =
            extension_features.shader_input_attachment_array_non_uniform_indexing;
        core_features.shader_uniform_texel_buffer_array_non_uniform_indexing =
            extension_features.shader_uniform_texel_buffer_array_non_uniform_indexing;
        core_features.shader_storage_texel_buffer_array_non_uniform_indexing =
            extension_features.shader_storage_texel_buffer_array_non_uniform_indexing;
        core_features.descriptor_binding_uniform_buffer_update_after_bind =
            extension_features.descriptor_binding_uniform_buffer_update_after_bind;
        core_features.descriptor_binding_sampled_image_update_after_bind =
            extension_features.descriptor_binding_sampled_image_update_after_bind;
        core_features.descriptor_binding_storage_image_update_after_bind =
            extension_features.descriptor_binding_storage_image_update_after_bind;
        core_features.descriptor_binding_storage_buffer_update_after_bind =
            extension_features.descriptor_binding_storage_buffer_update_after_bind;
        core_features.descriptor_binding_uniform_texel_buffer_update_after_bind =
            extension_features.descriptor_binding_uniform_texel_buffer_update_after_bind;
        core_features.descriptor_binding_storage_texel_buffer_update_after_bind =
            extension_features.descriptor_binding_storage_texel_buffer_update_after_bind;
        core_features.descriptor_binding_update_unused_while_pending =
            extension_features.descriptor_binding_update_unused_while_pending;
        core_features.descriptor_binding_partially_bound =
            extension_features.descriptor_binding_partially_bound;
        core_features.descriptor_binding_variable_descriptor_count =
            extension_features.descriptor_binding_variable_descriptor_count;
        core_features.runtime_descriptor_array = extension_features.runtime_descriptor_array;
    }

    fn copy_properties(
        extension_properties: &Self::ExtensionProperties,
        core_properties: &mut VulkanCoreProperties<Self::Core>,
    ) {
        core_properties.max_update_after_bind_descriptors_in_all_pools =
            extension_properties.max_update_after_bind_descriptors_in_all_pools;
        core_properties.shader_uniform_buffer_array_non_uniform_indexing_native =
            extension_properties.shader_uniform_buffer_array_non_uniform_indexing_native;
        core_properties.shader_sampled_image_array_non_uniform_indexing_native =
            extension_properties.shader_sampled_image_array_non_uniform_indexing_native;
        core_properties.shader_storage_buffer_array_non_uniform_indexing_native =
            extension_properties.shader_storage_buffer_array_non_uniform_indexing_native;
        core_properties.shader_storage_image_array_non_uniform_indexing_native =
            extension_properties.shader_storage_image_array_non_uniform_indexing_native;
        core_properties.shader_input_attachment_array_non_uniform_indexing_native =
            extension_properties.shader_input_attachment_array_non_uniform_indexing_native;
        core_properties.robust_buffer_access_update_after_bind =
            extension_properties.robust_buffer_access_update_after_bind;
        core_properties.quad_divergent_implicit_lod =
            extension_properties.quad_divergent_implicit_lod;
        core_properties.max_per_stage_descriptor_update_after_bind_samplers =
            extension_properties.max_per_stage_descriptor_update_after_bind_samplers;
        core_properties.max_per_stage_descriptor_update_after_bind_uniform_buffers =
            extension_properties.max_per_stage_descriptor_update_after_bind_uniform_buffers;
        core_properties.max_per_stage_descriptor_update_after_bind_storage_buffers =
            extension_properties.max_per_stage_descriptor_update_after_bind_storage_buffers;
        core_properties.max_per_stage_descriptor_update_after_bind_sampled_images =
            extension_properties.max_per_stage_descriptor_update_after_bind_sampled_images;
        core_properties.max_per_stage_descriptor_update_after_bind_storage_images =
            extension_properties.max_per_stage_descriptor_update_after_bind_storage_images;
        core_properties.max_per_stage_descriptor_update_after_bind_input_attachments =
            extension_properties.max_per_stage_descriptor_update_after_bind_input_attachments;
        core_properties.max_per_stage_update_after_bind_resources =
            extension_properties.max_per_stage_update_after_bind_resources;
        core_properties.max_descriptor_set_update_after_bind_samplers =
            extension_properties.max_descriptor_set_update_after_bind_samplers;
        core_properties.max_descriptor_set_update_after_bind_uniform_buffers =
            extension_properties.max_descriptor_set_update_after_bind_uniform_buffers;
        core_properties.max_descriptor_set_update_after_bind_uniform_buffers_dynamic =
            extension_properties.max_descriptor_set_update_after_bind_uniform_buffers_dynamic;
        core_properties.max_descriptor_set_update_after_bind_storage_buffers =
            extension_properties.max_descriptor_set_update_after_bind_storage_buffers;
        core_properties.max_descriptor_set_update_after_bind_storage_buffers_dynamic =
            extension_properties.max_descriptor_set_update_after_bind_storage_buffers_dynamic;
        core_properties.max_descriptor_set_update_after_bind_sampled_images =
            extension_properties.max_descriptor_set_update_after_bind_sampled_images;
        core_properties.max_descriptor_set_update_after_bind_storage_images =
            extension_properties.max_descriptor_set_update_after_bind_storage_images;
        core_properties.max_descriptor_set_update_after_bind_input_attachments =
            extension_properties.max_descriptor_set_update_after_bind_input_attachments;
    }

    fn process_features<L, I>(
        available: &DeviceFeatures,
        enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        let available = &available.v1_2;
        let enabled = enabled.get_mut();
        let mut changed = false;
        if required.remove(&DeviceFeature::DescriptorBindingSampledImageUpdateAfterBind) {
            assert!(
                available.shader_sampled_image_array_non_uniform_indexing != 0,
                "`DescriptorBindingSampledImageUpdateAfterBind` feature is required but not supported"
            );
            enabled.descriptor_binding_sampled_image_update_after_bind = 1;
            changed = true;
        }
        if required.remove(&DeviceFeature::DescriptorBindingStorageImageUpdateAfterBind) {
            assert!(
                available.shader_storage_image_array_non_uniform_indexing != 0,
                "`DescriptorBindingStorageImageUpdateAfterBind` feature is required but not supported"
            );
            enabled.descriptor_binding_storage_image_update_after_bind = 1;
            changed = true;
        }
        if required.remove(&DeviceFeature::DescriptorBindingUniformTexelBufferUpdateAfterBind) {
            assert!(
                available.shader_uniform_texel_buffer_array_non_uniform_indexing != 0,
                "`DescriptorBindingUniformTexelBufferUpdateAfterBind` feature is required but not supported"
            );
            enabled.descriptor_binding_uniform_texel_buffer_update_after_bind = 1;
            changed = true;
        }
        if required.remove(&DeviceFeature::DescriptorBindingStorageTexelBufferUpdateAfterBind) {
            assert!(
                available.shader_storage_texel_buffer_array_non_uniform_indexing != 0,
                "`DescriptorBindingStorageTexelBufferUpdateAfterBind` feature is required but not supported"
            );
            enabled.descriptor_binding_storage_texel_buffer_update_after_bind = 1;
            changed = true;
        }
        if required.remove(&DeviceFeature::DescriptorBindingUniformBufferUpdateAfterBind) {
            assert!(
                available.shader_uniform_buffer_array_non_uniform_indexing != 0,
                "`DescriptorBindingUniformBufferUpdateAfterBind` feature is required but not supported"
            );
            enabled.descriptor_binding_uniform_buffer_update_after_bind = 1;
            changed = true;
        }
        if required.remove(&DeviceFeature::DescriptorBindingStorageBufferUpdateAfterBind) {
            assert!(
                available.shader_storage_buffer_array_non_uniform_indexing != 0,
                "`DescriptorBindingStorageBufferUpdateAfterBind` feature is required but not supported"
            );
            enabled.descriptor_binding_storage_buffer_update_after_bind = 1;
            changed = true;
        }
        if required.remove(&DeviceFeature::DescriptorBindingPartiallyBound) {
            assert!(
                available.descriptor_binding_partially_bound != 0,
                "`DescriptorBindingPartiallyBound` feature is required but not supported"
            );
            enabled.descriptor_binding_partially_bound = 1;
            changed = true;
        }
        changed
    }
}

pub struct DisplayTimingExtension;

impl VulkanExtension for DisplayTimingExtension {
    const META: &'static vk::Extension = &vk::GOOGLE_DISPLAY_TIMING_EXTENSION;

    type Core = VulkanCoreUnknown;
    type ExtensionFeatures = NoFeatures;
    type ExtensionProperties = NoProperties;

    fn process_features<L, I>(
        _available: &DeviceFeatures,
        _enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        required.remove(&DeviceFeature::DisplayTiming)
    }
}

pub struct SamplerFilterMinMaxExtension;

impl VulkanExtension for SamplerFilterMinMaxExtension {
    const META: &'static vk::Extension = &vk::EXT_SAMPLER_FILTER_MINMAX_EXTENSION;

    type Core = VulkanCore<1, 2>;
    type ExtensionFeatures = NoFeatures;
    type ExtensionProperties = WithProperties<vk::PhysicalDeviceSamplerFilterMinmaxProperties>;

    fn copy_features(
        _extension_features: &Self::ExtensionFeatures,
        core_features: &mut DeviceFeatures,
    ) {
        core_features.v1_2.sampler_filter_minmax = 1;
    }

    fn copy_properties(
        extension_properties: &Self::ExtensionProperties,
        core_properties: &mut VulkanCoreProperties<Self::Core>,
    ) {
        core_properties.filter_minmax_single_component_formats =
            extension_properties.filter_minmax_single_component_formats;
        core_properties.filter_minmax_image_component_mapping =
            extension_properties.filter_minmax_image_component_mapping;
    }

    fn process_features<L, I>(
        available: &DeviceFeatures,
        _enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        let mut changed = false;
        if required.remove(&DeviceFeature::SamplerFilterMinMax) {
            assert!(
                available.v1_2.sampler_filter_minmax != 0,
                "`SamplerFilterMinMax` feature is required but not supported"
            );
            changed = true;
        }
        changed
    }
}

pub struct ScalarBlockLayoutExtension;

impl VulkanExtension for ScalarBlockLayoutExtension {
    const META: &'static vk::Extension = &vk::EXT_SCALAR_BLOCK_LAYOUT_EXTENSION;

    type Core = VulkanCore<1, 2>;
    type ExtensionFeatures = WithFeatures<vk::PhysicalDeviceScalarBlockLayoutFeatures>;
    type ExtensionProperties = NoProperties;

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut DeviceFeatures,
    ) {
        core_features.v1_2.scalar_block_layout = extension_features.scalar_block_layout;
    }

    fn process_features<L, I>(
        available: &DeviceFeatures,
        enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        let enabled = enabled.get_mut();
        let mut changed = false;
        if required.remove(&DeviceFeature::ScalarBlockLayout) {
            assert!(
                available.v1_2.scalar_block_layout != 0,
                "`ScalarBlockLayout` feature is required but not supported"
            );
            enabled.scalar_block_layout = 1;
            changed = true;
        }
        changed
    }
}

pub struct SurfacePresentationExtension;

impl VulkanExtension for SurfacePresentationExtension {
    const META: &'static vk::Extension = &vk::KHR_SWAPCHAIN_EXTENSION;

    type Core = VulkanCoreUnknown;
    type ExtensionFeatures = NoFeatures;
    type ExtensionProperties = NoProperties;

    fn process_features<L, I>(
        _available: &DeviceFeatures,
        _enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        required.remove(&DeviceFeature::SurfacePresentation)
    }
}

// === Stuff ===

pub trait ExtensionsHList: HList {
    type Features: HList;
    type Properties: HList;

    fn make_features() -> Self::Features;
    fn make_properties() -> Self::Properties;

    fn physical_device_properties2_push_all<'a, F>(
        api_version: u32,
        has_extension: F,
        builder: vk::PhysicalDeviceProperties2Builder<'a>,
        properties: &'a mut Self::Properties,
    ) -> vk::PhysicalDeviceProperties2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool;

    fn physical_device_features2_push_all<'a, F>(
        api_version: u32,
        has_extension: F,
        builder: vk::PhysicalDeviceFeatures2Builder<'a>,
        features: &'a mut Self::Features,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool;

    fn copy_features<F>(
        api_version: u32,
        has_extension: F,
        extension_features: &Self::Features,
        core_features: &mut DeviceFeatures,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool;
}

impl ExtensionsHList for HNil {
    type Features = HNil;
    type Properties = HNil;

    #[inline]
    fn make_features() -> Self::Features {
        HNil
    }

    #[inline]
    fn make_properties() -> Self::Properties {
        HNil
    }

    #[inline]
    fn physical_device_properties2_push_all<'a, F>(
        _api_version: u32,
        _has_extension: F,
        builder: vk::PhysicalDeviceProperties2Builder<'a>,
        _properties: &'a mut Self::Properties,
    ) -> vk::PhysicalDeviceProperties2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        builder
    }

    #[inline]
    fn physical_device_features2_push_all<'a, F>(
        _api_version: u32,
        _has_extension: F,
        builder: vk::PhysicalDeviceFeatures2Builder<'a>,
        _features: &'a mut Self::Features,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        builder
    }

    #[inline]
    fn copy_features<F>(
        _api_version: u32,
        _has_extension: F,
        _extension_features: &Self::Features,
        _core_features: &mut DeviceFeatures,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
    }
}

impl<H, T> ExtensionsHList for HCons<H, T>
where
    H: VulkanExtension,
    T: ExtensionsHList,
{
    type Features = HCons<H::ExtensionFeatures, T::Features>;
    type Properties = HCons<H::ExtensionProperties, T::Properties>;

    #[inline]
    fn make_features() -> Self::Features {
        HCons {
            head: H::ExtensionFeatures::default(),
            tail: T::make_features(),
        }
    }

    #[inline]
    fn make_properties() -> Self::Properties {
        HCons {
            head: H::ExtensionProperties::default(),
            tail: T::make_properties(),
        }
    }

    fn physical_device_properties2_push_all<'a, F>(
        api_version: u32,
        mut has_extension: F,
        mut builder: vk::PhysicalDeviceProperties2Builder<'a>,
        properties: &'a mut Self::Properties,
    ) -> vk::PhysicalDeviceProperties2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        if !H::Core::is_supported(api_version) && has_extension(H::META) {
            builder = properties
                .head
                .physical_device_properties2_push_next(builder);
        }
        T::physical_device_properties2_push_all(
            api_version,
            has_extension,
            builder,
            &mut properties.tail,
        )
    }

    #[inline]
    fn physical_device_features2_push_all<'a, F>(
        api_version: u32,
        mut has_extension: F,
        mut builder: vk::PhysicalDeviceFeatures2Builder<'a>,
        features: &'a mut Self::Features,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        if !H::Core::is_supported(api_version) && has_extension(H::META) {
            builder = features.head.physical_device_features2_push_next(builder);
        }
        T::physical_device_features2_push_all(
            api_version,
            has_extension,
            builder,
            &mut features.tail,
        )
    }

    fn copy_features<F>(
        api_version: u32,
        mut has_extension: F,
        extension_features: &Self::Features,
        core_features: &mut DeviceFeatures,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        if !H::Core::is_supported(api_version) && has_extension(H::META) {
            H::copy_features(&extension_features.head, core_features);
        }

        <T as ExtensionsHList>::copy_features(
            api_version,
            has_extension,
            &extension_features.tail,
            core_features,
        );
    }
}

pub trait VulkanExtensionsCollection {
    type Extensions: ExtensionsHList;

    fn make_features() -> <Self::Extensions as ExtensionsHList>::Features {
        Self::Extensions::make_features()
    }

    fn make_properties() -> <Self::Extensions as ExtensionsHList>::Properties {
        Self::Extensions::make_properties()
    }

    fn physical_device_properties2_push_all<'a, F>(
        api_version: u32,
        has_extension: F,
        builder: vk::PhysicalDeviceProperties2Builder<'a>,
        properties: &'a mut <Self::Extensions as ExtensionsHList>::Properties,
    ) -> vk::PhysicalDeviceProperties2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        <Self::Extensions as ExtensionsHList>::physical_device_properties2_push_all(
            api_version,
            has_extension,
            builder,
            properties,
        )
    }

    #[inline]
    fn physical_device_features2_push_all<'a, F>(
        api_version: u32,
        has_extension: F,
        builder: vk::PhysicalDeviceFeatures2Builder<'a>,
        features: &'a mut <Self::Extensions as ExtensionsHList>::Features,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        <Self::Extensions as ExtensionsHList>::physical_device_features2_push_all(
            api_version,
            has_extension,
            builder,
            features,
        )
    }
}

impl VulkanExtensionsCollection for () {
    type Extensions = HNil;
}

macro_rules! impl_vulkan_extensions_collection {
    ($($ty:ident),+$(,)?) => {
        impl<$($ty),*> VulkanExtensionsCollection for ($($ty),*,)
        where
            $($ty: VulkanExtension),*
        {
            type Extensions = <($($ty),*,) as TupleToHList>::HList;
        }
    };
}

impl_vulkan_extensions_collection!(T0);
impl_vulkan_extensions_collection!(T0, T1);
impl_vulkan_extensions_collection!(T0, T1, T2);
impl_vulkan_extensions_collection!(T0, T1, T2, T3);
impl_vulkan_extensions_collection!(T0, T1, T2, T3, T4);
impl_vulkan_extensions_collection!(T0, T1, T2, T3, T4, T5);
impl_vulkan_extensions_collection!(T0, T1, T2, T3, T4, T5, T6);
impl_vulkan_extensions_collection!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_vulkan_extensions_collection!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_vulkan_extensions_collection!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);

pub trait VulkanExtension {
    const META: &'static vk::Extension;

    type Core: VulkanCoreTypes;
    type ExtensionFeatures: VulkanFeatures;
    type ExtensionProperties: VulkanProperties;

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut DeviceFeatures,
    ) {
        _ = extension_features;
        _ = core_features;
    }

    fn copy_properties(
        extension_properties: &Self::ExtensionProperties,
        core_properties: &mut VulkanCoreProperties<Self::Core>,
    ) {
        _ = extension_properties;
        _ = core_properties;
    }

    fn process_features<L, I>(
        available: &DeviceFeatures,
        enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>;
}

pub struct InvalidExtension;

impl VulkanExtension for InvalidExtension {
    const META: &'static vk::Extension = &INVALID_EXTENSION;

    type Core = VulkanCoreUnknown;
    type ExtensionFeatures = NoFeatures;
    type ExtensionProperties = NoProperties;

    fn process_features<L, I>(_: &DeviceFeatures, _: &mut L, _: &mut HashSet<DeviceFeature>) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        unreachable!()
    }
}

const INVALID_EXTENSION: vk::Extension = vk::Extension {
    name: vk::ExtensionName::from_bytes(b"invalid"),
    number: 258,
    type_: "device",
    author: "invalid",
    contact: "invalid",
    platform: None,
    required_extensions: None,
    required_version: None,
    deprecated_by: None,
    obsoleted_by: None,
    promoted_to: None,
};

// === Core ===

pub trait VulkanCoreTypes {
    type Properties: Default;
    type Features: Default;

    fn is_supported(api_version: u32) -> bool;
}

pub type VulkanCoreProperties<T> = <T as VulkanCoreTypes>::Properties;
pub type VulkanCoreFeatures<T> = <T as VulkanCoreTypes>::Features;

pub struct VulkanCore<const MAJOR: u32, const MINOR: u32>;

impl<const MAJOR: u32, const MINOR: u32> VulkanCore<MAJOR, MINOR> {
    const API_VERSION: u32 = vk::make_version(MAJOR, MINOR, 0);
}

pub type VulkanCoreUnknown = VulkanCore<999, 999>;

impl VulkanCoreTypes for VulkanCoreUnknown {
    type Features = ();
    type Properties = ();

    fn is_supported(api_version: u32) -> bool {
        false
    }
}

impl VulkanCoreTypes for VulkanCore<1, 0> {
    type Features = vk::PhysicalDeviceFeatures;
    type Properties = vk::PhysicalDeviceProperties;

    fn is_supported(api_version: u32) -> bool {
        api_version >= Self::API_VERSION
    }
}

impl VulkanCoreTypes for VulkanCore<1, 1> {
    type Features = vk::PhysicalDeviceVulkan11Features;
    type Properties = vk::PhysicalDeviceVulkan11Properties;

    fn is_supported(api_version: u32) -> bool {
        api_version >= Self::API_VERSION
    }
}

impl VulkanCoreTypes for VulkanCore<1, 2> {
    type Features = vk::PhysicalDeviceVulkan12Features;
    type Properties = vk::PhysicalDeviceVulkan12Properties;

    fn is_supported(api_version: u32) -> bool {
        api_version >= Self::API_VERSION
    }
}

impl VulkanCoreTypes for VulkanCore<1, 3> {
    type Features = vk::PhysicalDeviceVulkan13Features;
    type Properties = vk::PhysicalDeviceVulkan13Properties;

    fn is_supported(api_version: u32) -> bool {
        api_version >= Self::API_VERSION
    }
}

// === Properties ===

pub trait VulkanProperties: Default {
    fn physical_device_properties2_push_next<'a>(
        &'a mut self,
        physical_device_properties2: vk::PhysicalDeviceProperties2Builder<'a>,
    ) -> vk::PhysicalDeviceProperties2Builder<'a>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoProperties;

impl VulkanProperties for NoProperties {
    #[inline]
    fn physical_device_properties2_push_next<'a>(
        &'a mut self,
        builder: vk::PhysicalDeviceProperties2Builder<'a>,
    ) -> vk::PhysicalDeviceProperties2Builder<'a> {
        builder
    }
}

#[repr(transparent)]
pub struct WithProperties<T>(pub T);

impl<T> WithProperties<T> {
    #[inline]
    fn wrap(inner: &T) -> &Self {
        // SAFETY: `WithProperties` is `repr(transparent)`
        unsafe { &*(inner as *const T).cast() }
    }

    #[inline]
    fn wrap_mut(inner: &mut T) -> &mut Self {
        // SAFETY: `WithProperties` is `repr(transparent)`
        unsafe { &mut *(inner as *mut T).cast() }
    }
}

impl<T: Default> Default for WithProperties<T> {
    #[inline]
    fn default() -> Self {
        Self(T::default())
    }
}

impl<T> std::ops::Deref for WithProperties<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for WithProperties<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> VulkanProperties for WithProperties<T>
where
    T: Default + vk::ExtendsPhysicalDeviceProperties2 + vk::Cast<Target = T>,
{
    #[inline]
    fn physical_device_properties2_push_next<'a>(
        &'a mut self,
        builder: vk::PhysicalDeviceProperties2Builder<'a>,
    ) -> vk::PhysicalDeviceProperties2Builder<'a> {
        builder.push_next::<T>(&mut self.0)
    }
}

// === Features ===

pub trait VulkanFeatures: Default {
    fn physical_device_features2_push_next<'a>(
        &'a mut self,
        builder: vk::PhysicalDeviceFeatures2Builder<'a>,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a>;

    fn device_create_info_push_next<'a>(
        &'a mut self,
        builder: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a>;
}

trait ExtractFeatures<T> {
    fn extract_features(&self) -> &T;
    fn extract_features_mut(&mut self) -> &mut T;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoFeatures;

impl VulkanFeatures for NoFeatures {
    #[inline]
    fn physical_device_features2_push_next<'a>(
        &'a mut self,
        builder: vk::PhysicalDeviceFeatures2Builder<'a>,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a> {
        builder
    }

    #[inline]
    fn device_create_info_push_next<'a>(
        &'a mut self,
        builder: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a> {
        builder
    }
}

impl<T> ExtractFeatures<NoFeatures> for T {
    #[inline]
    fn extract_features(&self) -> &NoFeatures {
        &NoFeatures
    }

    #[inline]
    fn extract_features_mut(&mut self) -> &mut NoFeatures {
        const _: () = assert!(std::mem::size_of::<NoFeatures>() == 0);

        // NOTE: does not allocate because `NoFeatures` is ZST
        // Related issue (return &mut to temp ZST): https://github.com/rust-lang/rust/issues/103821
        Box::leak(Box::new(NoFeatures))
    }
}

#[repr(transparent)]
pub struct WithFeatures<T>(pub T);

impl<T> WithFeatures<T> {
    #[inline]
    fn wrap(inner: &T) -> &Self {
        // SAFETY: `WithFeatures` is `repr(transparent)`
        unsafe { &*(inner as *const T).cast() }
    }

    #[inline]
    fn wrap_mut(inner: &mut T) -> &mut Self {
        // SAFETY: `WithFeatures` is `repr(transparent)`
        unsafe { &mut *(inner as *mut T).cast() }
    }
}

impl<T: Default> Default for WithFeatures<T> {
    #[inline]
    fn default() -> Self {
        Self(T::default())
    }
}

impl<T> std::ops::Deref for WithFeatures<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for WithFeatures<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, F> ExtractFeatures<WithFeatures<F>> for T
where
    T: AsRef<F> + AsMut<F>,
{
    #[inline]
    fn extract_features(&self) -> &WithFeatures<F> {
        WithFeatures::wrap(self.as_ref())
    }

    #[inline]
    fn extract_features_mut(&mut self) -> &mut WithFeatures<F> {
        WithFeatures::wrap_mut(self.as_mut())
    }
}

impl<T> VulkanFeatures for WithFeatures<T>
where
    T: Default
        + vk::ExtendsPhysicalDeviceFeatures2
        + vk::ExtendsDeviceCreateInfo
        + vk::Cast<Target = T>,
{
    #[inline]
    fn physical_device_features2_push_next<'a>(
        &'a mut self,
        builder: vk::PhysicalDeviceFeatures2Builder<'a>,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a> {
        builder.push_next(&mut self.0)
    }

    #[inline]
    fn device_create_info_push_next<'a>(
        &'a mut self,
        builder: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a> {
        builder.push_next::<T>(&mut self.0)
    }
}

// === HList ===

macro_rules! hlist_ty {
    ($($ty:ident),+) => { hlist_ty!(@inner [] [] $($ty)+) };

    (@inner [ $($prev:tt)* ] [ $($closing:tt)* ] $ty:ident) => {
        $($prev)* HCons<$ty, HNil>
        $($closing)*
    };
    (@inner [ $($prev:tt)* ] [ $($closing:tt)* ] $ty:ident $($rest:ident)+) => {
        hlist_ty!(@inner
            [$($prev)* HCons<$ty,]
            [$($closing)* >]
            $($rest)+
        )
    };
}

pub trait TupleToHList {
    type HList;

    fn into_hlist(self) -> Self::HList;
}

impl TupleToHList for () {
    type HList = HNil;

    #[inline]
    fn into_hlist(self) -> Self::HList {
        HNil
    }
}

macro_rules! impl_tuple_to_hlist {
    ($($idx:tt: $ty:ident),+$(,)?) => {
        impl<$($ty),*> TupleToHList for ($($ty),*,)
        {
            type HList = hlist_ty!($($ty),+);

            #[inline]
            fn into_hlist(self) -> Self::HList {
                impl_tuple_to_hlist!(@construct [ HNil ] [] $(self.$idx)+)
            }
        }
    };

    (@construct [ $($prev:tt)* ] []) => { $($prev)* };
    (@construct [ $($prev:tt)* ] [ $tuple:ident.$idx:tt $($rest:tt)* ]) => {
        impl_tuple_to_hlist!(@construct
            [HCons {
                head: $tuple.$idx,
                tail: $($prev)*,
            }]
            [ $($rest)* ]
        )
    };
    (@construct [ $($prev:tt)* ] [ $($reversed:tt)* ] $tuple:ident.$idx:tt $($rest:tt)* ) => {
        impl_tuple_to_hlist!(@construct
            [ $($prev)* ]
            [ $tuple.$idx $($reversed)*]
            $($rest)*
        )
    };
}

impl_tuple_to_hlist!(0: T0);
impl_tuple_to_hlist!(0: T0, 1: T1);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9);

pub trait HListToTuple {
    type Tuple;

    fn into_tuple(self) -> Self::Tuple;
}

impl HListToTuple for HNil {
    type Tuple = ();

    #[inline]
    fn into_tuple(self) -> Self::Tuple {}
}

macro_rules! impl_hlist_to_tuple {
    ($($ty:ident),+$(,)?) => {
        impl<$($ty),*> HListToTuple for hlist_ty!($($ty),+)
        {
            type Tuple = ($($ty),*,);

            #[inline]
            fn into_tuple(self) -> Self::Tuple {
                impl_hlist_to_tuple!(@deconstruct [] [self] $($ty)+)
            }
        }
    };

    (@deconstruct [ $($prev:tt)* ] [ $($prefix:tt)* ]) => { ($($prev)*) };
    (@deconstruct [ $($prev:tt)* ] [ $($prefix:tt)* ] $ty:ident $($rest:ident)*) => {
        impl_hlist_to_tuple!(@deconstruct
            [$($prev)* $($prefix)*.head,]
            [$($prefix)*.tail]
            $($rest)*
        )
    };
}

impl_hlist_to_tuple!(T0);
impl_hlist_to_tuple!(T0, T1);
impl_hlist_to_tuple!(T0, T1, T2);
impl_hlist_to_tuple!(T0, T1, T2, T3);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);

pub trait HList: Sized {
    fn prepend<H>(self, head: H) -> HCons<H, Self> {
        HCons { head, tail: self }
    }
}

#[derive(Debug, Default)]
pub struct HNil;

impl HList for HNil {}

impl AsRef<()> for HNil {
    #[inline]
    fn as_ref(&self) -> &() {
        &()
    }
}

impl AsMut<()> for HNil {
    #[inline]
    fn as_mut(&mut self) -> &mut () {
        Box::leak(Box::new(()))
    }
}

pub struct HCons<H, T: HList> {
    head: H,
    tail: T,
}

impl<H, T: HList> HList for HCons<H, T> {}

impl<H: Default, T: Default + HList> Default for HCons<H, T> {
    #[inline]
    fn default() -> Self {
        Self {
            head: H::default(),
            tail: T::default(),
        }
    }
}

impl<H: std::fmt::Debug, T: std::fmt::Debug + HList> std::fmt::Debug for HCons<H, T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("HCons")
            .field(&self.head)
            .field(&self.tail)
            .finish()
    }
}

pub trait Selector<S, I> {
    fn get(&self) -> &S;
    fn get_mut(&mut self) -> &mut S;
}

impl<T, Tail: HList> Selector<T, Here> for HCons<T, Tail> {
    #[inline]
    fn get(&self) -> &T {
        &self.head
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        &mut self.head
    }
}

impl<Head, Tail: HList, T, I> Selector<T, There<I>> for HCons<Head, Tail>
where
    Tail: Selector<T, I>,
{
    #[inline]
    fn get(&self) -> &T {
        self.tail.get()
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        self.tail.get_mut()
    }
}

enum Here {}

struct There<T> {
    _marker: std::marker::PhantomData<T>,
}
