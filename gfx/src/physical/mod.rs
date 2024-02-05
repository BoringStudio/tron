use shared::{FastHashMap, FastHashSet};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::InstanceV1_1;

use self::features::{AllExtensions, AllExtensionsExt, NoFeatures, NoProperties};
use crate::graphics::Graphics;
use crate::queue::{Queue, QueueFamily, QueueId, QueuesQuery};
use crate::types::{DeviceLost, OutOfDeviceMemory};
use crate::util::ToGfx;

pub use self::features::DeviceFeature;
pub use self::selector::{PhysicalDeviceSelector, PhysicalDeviceSelectorError};

mod features;
mod selector;

/// A wrapper around Vulkan physical device.
#[derive(Debug)]
pub struct PhysicalDevice {
    handle: vk::PhysicalDevice,
    properties: Box<DeviceProperties>,
    features: Box<DeviceFeatures>,
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
        let api_version = graphics.api_version();

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
        let mut require_extension = {
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

        let mut core_features = Box::<DeviceFeatures>::default();
        let mut extension_features = AllExtensions::make_features();

        // Fill extension features
        let mut min_api_version = vk::make_version(1, 0, 0);
        if api_version == min_api_version {
            let supported = require_extension(&vk::KHR_MAINTENANCE1_EXTENSION);
            assert!(
                supported,
                "`VK_KHR_maintenance1` extension must be supported"
            );
        }

        #[cfg(target_os = "macos")]
        if Graphics::requires_portability(api_version) {
            let supported = require_extension(&vk::KHR_PORTABILITY_SUBSET_EXTENSION);
            assert!(
                supported,
                "`VK_KHR_portability_subset` extension must be supported"
            );
        }

        device_create_info = AllExtensions::process_features(
            api_version,
            &mut min_api_version,
            require_extension,
            self.features.as_ref(),
            core_features.as_mut(),
            &mut extension_features,
            &mut requested_features,
            device_create_info,
        );

        // Fill core features
        if min_api_version >= vk::make_version(1, 3, 0) {
            device_create_info = device_create_info.push_next(&mut core_features.v1_3);
        }
        if min_api_version >= vk::make_version(1, 2, 0) {
            device_create_info = device_create_info.push_next(&mut core_features.v1_2);
        }
        if min_api_version >= vk::make_version(1, 1, 0) {
            device_create_info = device_create_info.push_next(&mut core_features.v1_1);
        }
        device_create_info = device_create_info.enabled_features(&core_features.v1_0);

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

        // === End fill feature requirements ===

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
            core_features,
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

macro_rules! impl_as_ref_mut(
    ($ident:ident, _: $rest:tt, $($field:ident: $ty:ty),*$(,)?) => {
        impl AsRef<$rest> for $ident {
            #[inline]
            fn as_ref(&self) -> &$rest {
                &$rest
            }
        }

        impl AsMut<$rest> for $ident {
            #[inline]
            fn as_mut(&mut self) -> &mut $rest {
                Box::leak(Box::new($rest))
            }
        }

        $(impl AsRef<$ty> for $ident {
            #[inline]
            fn as_ref(&self) -> &$ty {
                &self.$field
            }
        }

        impl AsMut<$ty> for $ident {
            #[inline]
            fn as_mut(&mut self) -> &mut $ty {
                &mut self.$field
            }
        })*
    };
);

/// All physical device properties.
#[derive(Debug, Default)]
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

impl_as_ref_mut!(
    DeviceProperties,
    _: NoProperties,
    v1_0: vk::PhysicalDeviceProperties,
    v1_1: vk::PhysicalDeviceVulkan11Properties,
    v1_2: vk::PhysicalDeviceVulkan12Properties,
    v1_3: vk::PhysicalDeviceVulkan13Properties,
);

/// All physical device features.
#[derive(Debug, Default)]
pub struct DeviceFeatures {
    pub v1_0: vk::PhysicalDeviceFeatures,
    pub v1_1: vk::PhysicalDeviceVulkan11Features,
    pub v1_2: vk::PhysicalDeviceVulkan12Features,
    pub v1_3: vk::PhysicalDeviceVulkan13Features,
}

unsafe impl Sync for DeviceFeatures {}
unsafe impl Send for DeviceFeatures {}

impl_as_ref_mut!(
    DeviceFeatures,
    _: NoFeatures,
    v1_0: vk::PhysicalDeviceFeatures,
    v1_1: vk::PhysicalDeviceVulkan11Features,
    v1_2: vk::PhysicalDeviceVulkan12Features,
    v1_3: vk::PhysicalDeviceVulkan13Features,
);

unsafe fn collect_info(handle: vk::PhysicalDevice) -> (Box<DeviceProperties>, Box<DeviceFeatures>) {
    let graphics = Graphics::get_unchecked();
    let instance = graphics.instance();
    let api_version = graphics.api_version();
    let (vk1_1, vk1_2, vk1_3) = (graphics.vk1_1(), graphics.vk1_2(), graphics.vk1_3());

    let extensions = instance
        .enumerate_device_extension_properties(handle, None)
        .unwrap()
        .into_iter()
        .map(|item| item.extension_name)
        .collect::<FastHashSet<_>>();
    let has_extension = |ext: &vk::Extension| -> bool { extensions.contains(&ext.name) };

    let mut core_features = Box::<DeviceFeatures>::default();
    let mut core_properties = Box::<DeviceProperties>::default();
    let mut properties_mt3 = vk::PhysicalDeviceMaintenance3Properties::builder();

    let mut extension_features = AllExtensions::make_features();
    let mut extension_properties = AllExtensions::make_properties();

    // Query info
    if vk1_1
        || instance
            .extensions()
            .contains(&vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name)
    {
        {
            let mut features2 = vk::PhysicalDeviceFeatures2::builder();
            let mut properties2 = vk::PhysicalDeviceProperties2::builder();

            // Core properties and features
            if vk1_1 {
                features2 = features2.push_next(&mut core_features.v1_1);
                properties2 = properties2.push_next(&mut core_properties.v1_1);
            }
            if vk1_2 {
                features2 = features2.push_next(&mut core_features.v1_2);
                properties2 = properties2.push_next(&mut core_properties.v1_2);
            }
            if vk1_3 {
                features2 = features2.push_next(&mut core_features.v1_3);
                properties2 = properties2.push_next(&mut core_properties.v1_3);
            }

            // Mandatory extension properties and features
            if !vk1_1 && has_extension(&vk::KHR_MAINTENANCE3_EXTENSION) {
                properties2 = properties2.push_next(&mut properties_mt3);
            }

            // Other extension properties and features
            features2 = AllExtensions::physical_device_features2_push_all(
                api_version,
                has_extension,
                features2,
                &mut extension_features,
            );
            properties2 = AllExtensions::physical_device_properties2_push_all(
                api_version,
                has_extension,
                properties2,
                &mut extension_properties,
            );

            // Query extended info
            instance.get_physical_device_features2(handle, &mut features2);
            instance.get_physical_device_properties2(handle, &mut properties2);

            core_features.v1_0 = features2.features;
            core_properties.v1_0 = properties2.properties;
        }

        core_features.v1_3.next = std::ptr::null_mut();
        core_features.v1_2.next = std::ptr::null_mut();
        core_features.v1_1.next = std::ptr::null_mut();

        properties_mt3.next = std::ptr::null_mut();
        core_properties.v1_3.next = std::ptr::null_mut();
        core_properties.v1_2.next = std::ptr::null_mut();
        core_properties.v1_1.next = std::ptr::null_mut();
    } else {
        // Query basic info
        core_features.v1_0 = instance.get_physical_device_features(handle);
        core_properties.v1_0 = instance.get_physical_device_properties(handle);
    }

    // Other info
    core_properties.queue_families = instance.get_physical_device_queue_family_properties(handle);
    core_properties.memory = instance.get_physical_device_memory_properties(handle);

    // Map mandatory extensions to core
    if !vk1_1 && has_extension(&vk::KHR_MAINTENANCE3_EXTENSION) {
        core_properties.v1_1.max_per_set_descriptors = properties_mt3.max_per_set_descriptors;
        core_properties.v1_1.max_memory_allocation_size = properties_mt3.max_memory_allocation_size;
    }

    // Map other extensions to core
    AllExtensions::copy_features(
        api_version,
        has_extension,
        &extension_features,
        core_features.as_mut(),
    );

    AllExtensions::copy_properties(
        api_version,
        has_extension,
        &extension_properties,
        core_properties.as_mut(),
    );

    // Done
    core_properties.extensions = extensions;
    (core_properties, core_features)
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
