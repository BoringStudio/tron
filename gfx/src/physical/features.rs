use shared::hlist::*;
use shared::FastHashSet;
use vulkanalia::prelude::v1_0::*;

/// A feature that can be requested when creating a device.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DeviceFeature {
    /// Adds a buffer device address to the [`Buffer`].
    ///
    /// [`Buffer`]: crate::Buffer
    BufferDeviceAddress,

    /// Allows using dynamic indexes for accessing arrays of
    /// type [`DescriptorType::SampledImage`].
    ///
    /// [`DescriptorType::SampledImage`]: crate::DescriptorType::SampledImage
    ShaderSampledImageDynamicIndexing,

    /// Allows using dynamic indexes for accessing arrays of
    /// type [`DescriptorType::StorageImage`].
    ///
    /// [`DescriptorType::StorageImage`]: crate::DescriptorType::StorageImage
    ShaderStorageImageDynamicIndexing,

    /// Allows using dynamic indexes for accessing arrays of
    /// type [`DescriptorType::UniformBuffer`].
    ///
    /// [`DescriptorType::UniformBuffer`]: crate::DescriptorType::UniformBuffer
    ShaderUniformBufferDynamicIndexing,

    /// Allows using dynamic indexes for accessing arrays of
    /// type [`DescriptorType::StorageBuffer`].
    ///
    /// [`DescriptorType::StorageBuffer`]: crate::DescriptorType::StorageBuffer
    ShaderStorageBufferDynamicIndexing,

    /// Allows using non-uniform indexes for accessing arrays of
    /// type [`DescriptorType::SampledImage`].
    ///
    /// [`DescriptorType::SampledImage`]: crate::DescriptorType::SampledImage
    ShaderSampledImageNonUniformIndexing,

    /// Allows using non-uniform indexes for accessing arrays of
    /// type [`DescriptorType::StorageImage`].
    ///
    /// [`DescriptorType::StorageImage`]: crate::DescriptorType::StorageImage
    ShaderStorageImageNonUniformIndexing,

    /// Allows using non-uniform indexes for accessing arrays of
    /// type [`DescriptorType::UniformBuffer`].
    ///
    /// [`DescriptorType::UniformBuffer`]: crate::DescriptorType::UniformBuffer
    ShaderUniformBufferNonUniformIndexing,

    /// Allows using non-uniform indexes for accessing arrays of
    /// type [`DescriptorType::StorageBuffer`].
    ///
    /// [`DescriptorType::StorageBuffer`]: crate::DescriptorType::StorageBuffer
    ShaderStorageBufferNonUniformIndexing,

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

    // Adds ability to declare descriptors in runtime arrays.
    RuntimeDescriptorArray,

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

impl DeviceFeature {
    #[track_caller]
    fn check(&self, required: &mut FastHashSet<DeviceFeature>, supported: bool) -> bool {
        let required = required.remove(self);
        if required {
            assert!(supported, "`{self:?}` is required but not supported");
        }
        required
    }
}

macro_rules! process_features {
    (
        { $available:ident, $enabled:ident, $required:ident },
        $($ident:ident => $field:ident),*$(,)?
    ) => {{
        let mut __changed = false;
        $(
            if DeviceFeature::$ident.check($required, $available.$field != 0) {
                $enabled.$field = 1;
                __changed = true;
            }
        )*
        __changed
    }};
}

pub type AllExtensions = (
    BaseExtension,
    BufferDeviceAddressExtension,
    DescriptorIndexingExtension,
    DisplayTimingExtension,
    SamplerFilterMinMaxExtension,
    ScalarBlockLayoutExtension,
    SurfacePresentationExtension,
);

/// Base Vulkan features.
pub struct BaseExtension;

impl VulkanExtension for BaseExtension {
    const META: &'static vk::Extension = &vk::Extension {
        name: vk::ExtensionName::from_bytes(b"no extension"),
        number: 0,
        type_: "device",
        author: "",
        contact: "",
        platform: None,
        required_extensions: None,
        required_version: None,
        deprecated_by: None,
        obsoleted_by: None,
        promoted_to: None,
    };

    type Core = VulkanCore<1, 0>;
    type ExtensionFeatures = WithFeatures<BaseFeatures>;
    type ExtensionProperties = NoProperties;

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut VulkanCoreFeatures<Self::Core>,
    ) {
        core_features.shader_sampled_image_array_dynamic_indexing =
            extension_features.shader_sampled_image_array_dynamic_indexing;
        core_features.shader_storage_image_array_dynamic_indexing =
            extension_features.shader_storage_image_array_dynamic_indexing;
        core_features.shader_uniform_buffer_array_dynamic_indexing =
            extension_features.shader_uniform_buffer_array_dynamic_indexing;
        core_features.shader_storage_buffer_array_dynamic_indexing =
            extension_features.shader_storage_buffer_array_dynamic_indexing;
    }

    fn process_features(
        available: &VulkanCoreFeatures<Self::Core>,
        enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool {
        process_features!(
            { available, enabled, required },
            ShaderSampledImageDynamicIndexing => shader_sampled_image_array_dynamic_indexing,
            ShaderStorageImageDynamicIndexing => shader_storage_image_array_dynamic_indexing,
            ShaderUniformBufferDynamicIndexing => shader_uniform_buffer_array_dynamic_indexing,
            ShaderStorageBufferDynamicIndexing => shader_storage_buffer_array_dynamic_indexing,
        )
    }
}

#[derive(Debug, Default)]
pub struct BaseFeatures {
    shader_sampled_image_array_dynamic_indexing: vk::Bool32,
    shader_storage_image_array_dynamic_indexing: vk::Bool32,
    shader_uniform_buffer_array_dynamic_indexing: vk::Bool32,
    shader_storage_buffer_array_dynamic_indexing: vk::Bool32,
}

unsafe impl vk::Cast for BaseFeatures {
    type Target = Self;

    #[inline]
    fn into(self) -> Self::Target {
        panic!("must never be called");
    }
}

// SAFETY: `BaseExtension` is always supported by core, so it will never be
// passed to `...::push_next`, or it will at least panic if it is.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for BaseFeatures {}
unsafe impl vk::ExtendsDeviceCreateInfo for BaseFeatures {}

pub struct BufferDeviceAddressExtension;

impl VulkanExtension for BufferDeviceAddressExtension {
    const META: &'static vk::Extension = &vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION;

    type Core = VulkanCore<1, 2>;
    type ExtensionFeatures = WithFeatures<vk::PhysicalDeviceBufferDeviceAddressFeatures>;
    type ExtensionProperties = NoProperties;

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut VulkanCoreFeatures<Self::Core>,
    ) {
        core_features.buffer_device_address = extension_features.buffer_device_address;
        core_features.buffer_device_address_capture_replay =
            extension_features.buffer_device_address_capture_replay;
        core_features.buffer_device_address_multi_device =
            extension_features.buffer_device_address_multi_device;
    }

    fn process_features(
        available: &VulkanCoreFeatures<Self::Core>,
        enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool {
        process_features!(
            { available, enabled, required },
            BufferDeviceAddress => buffer_device_address,
        )
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
        core_features: &mut VulkanCoreFeatures<Self::Core>,
    ) {
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

    fn process_features(
        available: &VulkanCoreFeatures<Self::Core>,
        enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool {
        process_features!(
            { available, enabled, required },
            ShaderSampledImageNonUniformIndexing => shader_sampled_image_array_non_uniform_indexing,
            ShaderStorageImageNonUniformIndexing => shader_storage_image_array_non_uniform_indexing,
            ShaderUniformBufferNonUniformIndexing => shader_uniform_buffer_array_non_uniform_indexing,
            ShaderStorageBufferNonUniformIndexing => shader_storage_buffer_array_non_uniform_indexing,
            DescriptorBindingSampledImageUpdateAfterBind => descriptor_binding_sampled_image_update_after_bind,
            DescriptorBindingStorageImageUpdateAfterBind => descriptor_binding_storage_image_update_after_bind,
            DescriptorBindingUniformTexelBufferUpdateAfterBind => descriptor_binding_uniform_texel_buffer_update_after_bind,
            DescriptorBindingStorageTexelBufferUpdateAfterBind => descriptor_binding_storage_texel_buffer_update_after_bind,
            DescriptorBindingUniformBufferUpdateAfterBind => descriptor_binding_uniform_buffer_update_after_bind,
            DescriptorBindingStorageBufferUpdateAfterBind => descriptor_binding_storage_buffer_update_after_bind,
            DescriptorBindingPartiallyBound => descriptor_binding_partially_bound,
            RuntimeDescriptorArray => runtime_descriptor_array,
        )
    }
}

pub struct DisplayTimingExtension;

impl VulkanExtension for DisplayTimingExtension {
    const META: &'static vk::Extension = &vk::GOOGLE_DISPLAY_TIMING_EXTENSION;

    type Core = VulkanCoreUnknown;
    type ExtensionFeatures = NoFeatures;
    type ExtensionProperties = NoProperties;

    fn process_features(
        _available: &VulkanCoreFeatures<Self::Core>,
        _enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool {
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
        core_features: &mut VulkanCoreFeatures<Self::Core>,
    ) {
        core_features.sampler_filter_minmax = 1;
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

    fn process_features(
        available: &VulkanCoreFeatures<Self::Core>,
        _enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool {
        DeviceFeature::SamplerFilterMinMax.check(required, available.sampler_filter_minmax != 0)
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
        core_features: &mut VulkanCoreFeatures<Self::Core>,
    ) {
        core_features.scalar_block_layout = extension_features.scalar_block_layout;
    }

    fn process_features(
        available: &VulkanCoreFeatures<Self::Core>,
        enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool {
        process_features!(
            { available, enabled, required },
            ScalarBlockLayout => scalar_block_layout,
        )
    }
}

pub struct SurfacePresentationExtension;

impl VulkanExtension for SurfacePresentationExtension {
    const META: &'static vk::Extension = &vk::KHR_SWAPCHAIN_EXTENSION;

    type Core = VulkanCoreUnknown;
    type ExtensionFeatures = NoFeatures;
    type ExtensionProperties = NoProperties;

    fn process_features(
        _available: &VulkanCoreFeatures<Self::Core>,
        _enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool {
        required.remove(&DeviceFeature::SurfacePresentation)
    }
}

// === Stuff ===

pub trait AllExtensionsExt {
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

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn process_features<'a, F, C>(
        api_version: u32,
        min_api_version: &mut u32,
        require_extension: F,
        available_core_features: &C,
        enabled_core_features: &mut C,
        enabled_extension_features: &'a mut <Self::Extensions as ExtensionsHList>::Features,
        required: &mut FastHashSet<DeviceFeature>,
        builder: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
        Self::Extensions: ExtensionsHListProcessFeatures<C>,
    {
        <Self::Extensions as ExtensionsHListProcessFeatures<C>>::process_features(
            api_version,
            min_api_version,
            require_extension,
            available_core_features,
            enabled_core_features,
            enabled_extension_features,
            required,
            builder,
        )
    }

    #[inline]
    fn copy_features<F, C>(
        api_version: u32,
        has_extension: F,
        extension_features: &<Self::Extensions as ExtensionsHList>::Features,
        core_features: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
        Self::Extensions: ExtensionsHListCopyFeatures<C>,
    {
        <Self::Extensions as ExtensionsHListCopyFeatures<C>>::copy_features(
            api_version,
            has_extension,
            extension_features,
            core_features,
        );
    }

    #[inline]
    fn copy_properties<F, C>(
        api_version: u32,
        has_extension: F,
        extension_properties: &<Self::Extensions as ExtensionsHList>::Properties,
        core_properties: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
        Self::Extensions: ExtensionsHListCopyProperties<C>,
    {
        <Self::Extensions as ExtensionsHListCopyProperties<C>>::copy_properties(
            api_version,
            has_extension,
            extension_properties,
            core_properties,
        );
    }
}

impl AllExtensionsExt for () {
    type Extensions = HNil;
}

macro_rules! impl_vulkan_extensions_collection {
    ($($ty:ident),+$(,)?) => {
        impl<$($ty),*> AllExtensionsExt for ($($ty),*,)
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
}

//

pub trait ExtensionsHListProcessFeatures<C>: ExtensionsHList {
    #[allow(clippy::too_many_arguments)]
    fn process_features<'a, F>(
        api_version: u32,
        min_api_version: &mut u32,
        require_extension: F,
        available_core_features: &C,
        enabled_core_features: &mut C,
        enabled_extension_features: &'a mut Self::Features,
        required: &mut FastHashSet<DeviceFeature>,
        builder: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool;
}

impl<C> ExtensionsHListProcessFeatures<C> for HNil {
    #[inline]
    fn process_features<'a, F>(
        _api_version: u32,
        _min_api_version: &mut u32,
        _require_extension: F,
        _available_core_features: &C,
        _enabled_core_features: &mut C,
        _enabled_extension_features: &'a mut Self::Features,
        _required: &mut FastHashSet<DeviceFeature>,
        builder: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        builder
    }
}

impl<C, H, T> ExtensionsHListProcessFeatures<C> for HCons<H, T>
where
    H: VulkanExtension,
    T: ExtensionsHList + ExtensionsHListProcessFeatures<C>,
    C: AsRef<VulkanCoreFeatures<H::Core>> + AsMut<VulkanCoreFeatures<H::Core>>,
{
    fn process_features<'a, F>(
        api_version: u32,
        min_api_version: &mut u32,
        mut require_extension: F,
        available_core_features: &C,
        enabled_core_features: &mut C,
        enabled_extension_features: &'a mut Self::Features,
        required: &mut FastHashSet<DeviceFeature>,
        mut builder: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a>
    where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        if H::process_features(
            available_core_features.as_ref(),
            &mut enabled_extension_features.head,
            required,
        ) {
            if H::Core::is_supported(api_version) {
                H::copy_features(
                    &enabled_extension_features.head,
                    enabled_core_features.as_mut(),
                );
                *min_api_version = std::cmp::max(*min_api_version, H::Core::API_VERSION);
            } else {
                let supported = require_extension(H::META);
                assert!(
                    supported,
                    "`{}` extension is required but not supported",
                    H::META.name
                );

                builder = enabled_extension_features
                    .head
                    .device_create_info_push_next(builder);
            }
        }

        <T as ExtensionsHListProcessFeatures<C>>::process_features(
            api_version,
            min_api_version,
            require_extension,
            available_core_features,
            enabled_core_features,
            &mut enabled_extension_features.tail,
            required,
            builder,
        )
    }
}

//

pub trait ExtensionsHListCopyFeatures<C>: ExtensionsHList {
    fn copy_features<F>(
        api_version: u32,
        has_extension: F,
        extension_features: &Self::Features,
        core_features: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool;
}

impl<C> ExtensionsHListCopyFeatures<C> for HNil {
    fn copy_features<F>(
        _api_version: u32,
        _has_extension: F,
        _extension_features: &Self::Features,
        _core_features: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
    }
}

impl<C, H, T> ExtensionsHListCopyFeatures<C> for HCons<H, T>
where
    H: VulkanExtension,
    T: ExtensionsHList + ExtensionsHListCopyFeatures<C>,
    C: AsMut<VulkanCoreFeatures<H::Core>>,
{
    #[inline]
    fn copy_features<F>(
        api_version: u32,
        mut has_extension: F,
        extension_features: &Self::Features,
        core_features: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        if !H::Core::is_supported(api_version) && has_extension(H::META) {
            H::copy_features(&extension_features.head, core_features.as_mut());
        }

        <T as ExtensionsHListCopyFeatures<C>>::copy_features(
            api_version,
            has_extension,
            &extension_features.tail,
            core_features,
        );
    }
}

//

pub trait ExtensionsHListCopyProperties<C>: ExtensionsHList {
    fn copy_properties<F>(
        api_version: u32,
        has_extension: F,
        extension_properties: &Self::Properties,
        core_properties: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool;
}

impl<C> ExtensionsHListCopyProperties<C> for HNil {
    fn copy_properties<F>(
        _api_version: u32,
        _has_extension: F,
        _extension_properties: &Self::Properties,
        _core_properties: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
    }
}

impl<C, H, T> ExtensionsHListCopyProperties<C> for HCons<H, T>
where
    H: VulkanExtension,
    T: ExtensionsHList + ExtensionsHListCopyProperties<C>,
    C: AsMut<VulkanCoreProperties<H::Core>>,
{
    #[inline]
    fn copy_properties<F>(
        api_version: u32,
        mut has_extension: F,
        extension_properties: &Self::Properties,
        core_properties: &mut C,
    ) where
        for<'e> F: FnMut(&'e vk::Extension) -> bool,
    {
        if !H::Core::is_supported(api_version) && has_extension(H::META) {
            H::copy_properties(&extension_properties.head, core_properties.as_mut());
        }

        <T as ExtensionsHListCopyProperties<C>>::copy_properties(
            api_version,
            has_extension,
            &extension_properties.tail,
            core_properties,
        );
    }
}

//

pub trait VulkanExtension {
    const META: &'static vk::Extension;

    type Core: VulkanCoreTypes;
    type ExtensionFeatures: VulkanFeatures;
    type ExtensionProperties: VulkanProperties;

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut VulkanCoreFeatures<Self::Core>,
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

    fn process_features(
        available: &VulkanCoreFeatures<Self::Core>,
        enabled: &mut Self::ExtensionFeatures,
        required: &mut FastHashSet<DeviceFeature>,
    ) -> bool;
}

// === Core ===

pub trait VulkanCoreTypes {
    const API_VERSION: u32;
    type Properties: Default;
    type Features: Default;

    fn is_supported(api_version: u32) -> bool {
        api_version >= Self::API_VERSION
    }
}

pub type VulkanCoreProperties<T> = <T as VulkanCoreTypes>::Properties;
pub type VulkanCoreFeatures<T> = <T as VulkanCoreTypes>::Features;

pub struct VulkanCore<const MAJOR: u32, const MINOR: u32>;

impl<const MAJOR: u32, const MINOR: u32> VulkanCore<MAJOR, MINOR> {
    pub const MAJOR: u32 = MAJOR;
    pub const MINOR: u32 = MINOR;
}

pub type VulkanCoreUnknown = VulkanCore<999, 999>;

impl VulkanCoreTypes for VulkanCoreUnknown {
    const API_VERSION: u32 = vk::make_version(Self::MAJOR, Self::MINOR, 0);
    type Features = NoFeatures;
    type Properties = NoProperties;

    #[inline]
    fn is_supported(_: u32) -> bool {
        false
    }
}

impl VulkanCoreTypes for VulkanCore<1, 0> {
    const API_VERSION: u32 = vk::make_version(Self::MAJOR, Self::MINOR, 0);
    type Features = vk::PhysicalDeviceFeatures;
    type Properties = vk::PhysicalDeviceProperties;
}

impl VulkanCoreTypes for VulkanCore<1, 1> {
    const API_VERSION: u32 = vk::make_version(Self::MAJOR, Self::MINOR, 0);
    type Features = vk::PhysicalDeviceVulkan11Features;
    type Properties = vk::PhysicalDeviceVulkan11Properties;
}

impl VulkanCoreTypes for VulkanCore<1, 2> {
    const API_VERSION: u32 = vk::make_version(Self::MAJOR, Self::MINOR, 0);
    type Features = vk::PhysicalDeviceVulkan12Features;
    type Properties = vk::PhysicalDeviceVulkan12Properties;
}

impl VulkanCoreTypes for VulkanCore<1, 3> {
    const API_VERSION: u32 = vk::make_version(Self::MAJOR, Self::MINOR, 0);
    type Features = vk::PhysicalDeviceVulkan13Features;
    type Properties = vk::PhysicalDeviceVulkan13Properties;
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

#[repr(transparent)]
pub struct WithFeatures<T>(pub T);

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
