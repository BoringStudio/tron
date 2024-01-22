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

pub struct BufferDeviceAddressExtension;

impl VulkanExtension for BufferDeviceAddressExtension {
    const META: &'static vk::Extension = &vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION;

    type Core = VulkanCore<1, 2>;
    type ExtensionFeatures = WithFeatures<vk::PhysicalDeviceBufferDeviceAddressFeatures>;
    type ExtensionProperties = ();

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

    fn process_features<L, I>(
        availabe: &VulkanCoreFeatures<Self::Core>,
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
                availabe.buffer_device_address != 0,
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
    type ExtensionProperties = vk::PhysicalDeviceDescriptorIndexingProperties;

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

    fn process_features<L, I>(
        available: &VulkanCoreFeatures<Self::Core>,
        enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
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
    type ExtensionProperties = ();

    fn process_features<L, I>(
        _available: &VulkanCoreFeatures<Self::Core>,
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
    type ExtensionProperties = vk::PhysicalDeviceSamplerFilterMinmaxProperties;

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

    fn process_features<L, I>(
        available: &VulkanCoreFeatures<Self::Core>,
        _enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>,
    {
        let mut changed = false;
        if required.remove(&DeviceFeature::SamplerFilterMinMax) {
            assert!(
                available.sampler_filter_minmax != 0,
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
    type ExtensionProperties = ();

    fn copy_features(
        extension_features: &Self::ExtensionFeatures,
        core_features: &mut VulkanCoreFeatures<Self::Core>,
    ) {
        core_features.scalar_block_layout = extension_features.scalar_block_layout;
    }

    fn process_features<L, I>(
        available: &VulkanCoreFeatures<Self::Core>,
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
                available.scalar_block_layout != 0,
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
    type ExtensionProperties = ();

    fn process_features<L, I>(
        _available: &VulkanCoreFeatures<Self::Core>,
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

pub trait VulkanExtensionsCollection {
    type Features: Default;
    type Properties: Default;
}

impl VulkanExtensionsCollection for () {
    type Features = ();
    type Properties = ();
}

macro_rules! impl_vulkan_extensions_collection {
    ($($ty:ident),*$(,)?) => {
        impl<$($ty),*> VulkanExtensionsCollection for ($($ty),*,)
        where
            $($ty: VulkanExtension),*
        {
            type Features = ($(<$ty>::ExtensionFeatures),*,);
            type Properties = ($(<$ty>::ExtensionProperties),*,);
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
    type ExtensionProperties: Default;

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

    fn process_features<L, I>(
        available: &VulkanCoreFeatures<Self::Core>,
        enabled: &mut L,
        required: &mut HashSet<DeviceFeature>,
    ) -> bool
    where
        L: Selector<Self::ExtensionFeatures, I>;
}

pub trait VulkanCoreTypes {
    type Properties: Default;
    type Features: Default;
}

pub type VulkanCoreProperties<T> = <T as VulkanCoreTypes>::Properties;
pub type VulkanCoreFeatures<T> = <T as VulkanCoreTypes>::Features;

pub struct VulkanCore<const MAJOR: u32, const MINOR: u32>;

impl<const MAJOR: u32, const MINOR: u32> VulkanCore<MAJOR, MINOR> {
    pub const fn is_supported(api_version: u32) -> bool {
        api_version >= vk::make_version(MAJOR, MINOR, 0)
    }
}

pub type VulkanCoreUnknown = VulkanCore<999, 999>;

impl VulkanCoreTypes for VulkanCoreUnknown {
    type Features = ();
    type Properties = ();
}

impl VulkanCoreTypes for VulkanCore<1, 0> {
    type Features = vk::PhysicalDeviceFeatures;
    type Properties = vk::PhysicalDeviceProperties;
}

impl VulkanCoreTypes for VulkanCore<1, 1> {
    type Features = vk::PhysicalDeviceVulkan11Features;
    type Properties = vk::PhysicalDeviceVulkan11Properties;
}

impl VulkanCoreTypes for VulkanCore<1, 2> {
    type Features = vk::PhysicalDeviceVulkan12Features;
    type Properties = vk::PhysicalDeviceVulkan12Properties;
}

impl VulkanCoreTypes for VulkanCore<1, 3> {
    type Features = vk::PhysicalDeviceVulkan13Features;
    type Properties = vk::PhysicalDeviceVulkan13Properties;
}

pub trait VulkanFeatures: Default {
    fn physical_device_features2_push_next<'a>(
        &'a mut self,
        physical_device_features2: vk::PhysicalDeviceFeatures2Builder<'a>,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a>;

    fn device_create_info_push_next<'a>(
        &'a mut self,
        device_create_info: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoFeatures;

impl VulkanFeatures for NoFeatures {
    #[inline]
    fn physical_device_features2_push_next<'a>(
        &'a mut self,
        physical_device_features2: vk::PhysicalDeviceFeatures2Builder<'a>,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a> {
        physical_device_features2
    }

    #[inline]
    fn device_create_info_push_next<'a>(
        &'a mut self,
        device_create_info: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a> {
        device_create_info
    }
}

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
        physical_device_features2: vk::PhysicalDeviceFeatures2Builder<'a>,
    ) -> vk::PhysicalDeviceFeatures2Builder<'a> {
        physical_device_features2.push_next(&mut self.0)
    }

    #[inline]
    fn device_create_info_push_next<'a>(
        &'a mut self,
        device_create_info: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a> {
        device_create_info.push_next::<T>(&mut self.0)
    }
}

// === HList ===

trait HList: Sized {
    fn prepend<H>(self, head: H) -> HCons<H, Self> {
        HCons { head, tail: self }
    }
}

struct HNil;

impl HList for HNil {}

struct HCons<H, T> {
    head: H,
    tail: T,
}

impl<H, T> HList for HCons<H, T> {}

pub trait Selector<S, I> {
    fn get(&self) -> &S;
    fn get_mut(&mut self) -> &mut S;
}

impl<T, Tail> Selector<T, Here> for HCons<T, Tail> {
    #[inline]
    fn get(&self) -> &T {
        &self.head
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        &mut self.head
    }
}

impl<Head, Tail, T, I> Selector<T, There<I>> for HCons<Head, Tail>
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
