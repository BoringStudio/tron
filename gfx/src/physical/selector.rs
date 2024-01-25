use shared::{FastHashMap, FastHashSet};
use vulkanalia::prelude::v1_0::*;

use super::features::DeviceFeature;
use super::CreateDeviceError;
use crate::queue::QueuesQuery;

/// A builder for selecting a physical device.
pub struct PhysicalDeviceSelector {
    physical_devices: Vec<gfx::PhysicalDevice>,
    requested_features: FastHashMap<DeviceFeature, Necessity>,
    allow_discrete_gpu: bool,
    allow_integrated_gpu: bool,
    allow_virtual_gpu: bool,
    allow_cpu: bool,
}

impl PhysicalDeviceSelector {
    pub(crate) fn new(physical_devices: Vec<gfx::PhysicalDevice>) -> Self {
        Self {
            physical_devices,
            requested_features: FastHashMap::default(),
            allow_discrete_gpu: true,
            allow_integrated_gpu: true,
            allow_virtual_gpu: true,
            allow_cpu: false,
        }
    }

    pub fn physical_devices(&self) -> &[gfx::PhysicalDevice] {
        &self.physical_devices
    }

    pub fn allow_discrete_gpu(mut self, allow: bool) -> Self {
        self.allow_discrete_gpu = allow;
        self
    }

    pub fn allow_integrated_gpu(mut self, allow: bool) -> Self {
        self.allow_integrated_gpu = allow;
        self
    }

    pub fn allow_virtual_gpu(mut self, allow: bool) -> Self {
        self.allow_virtual_gpu = allow;
        self
    }

    pub fn allow_cpu(mut self, allow: bool) -> Self {
        self.allow_cpu = allow;
        self
    }

    pub fn with_required_feature(mut self, feature: DeviceFeature) -> Self {
        self.requested_features.insert(feature, Necessity::Required);
        self
    }

    pub fn with_required_features(mut self, features: &[DeviceFeature]) -> Self {
        for feature in features {
            self.requested_features
                .insert(*feature, Necessity::Required);
        }
        self
    }

    // pub fn with_optional_feature(mut self, feature: DeviceFeature, score: usize) -> Self {
    //     self.requested_features
    //         .insert(feature, Necessity::Optional { score });
    //     self
    // }

    // pub fn with_optional_features(mut self, features: &[(DeviceFeature, usize)]) -> Self {
    //     for (feature, score) in features {
    //         self.requested_features
    //             .insert(*feature, Necessity::Optional { score: *score });
    //     }
    //     self
    // }

    // TODO: Add support for optional features
    pub fn find_best(mut self) -> Result<SelectedPhysicalDevice, PhysicalDeviceSelectorError> {
        let mut result = None;

        for (index, physical_device) in self.physical_devices.iter().enumerate() {
            let properties = physical_device.properties();

            tracing::info!(
                name = %properties.v1_0.device_name,
                ty = ?properties.v1_0.device_type,
                "found physical device",
            );

            let mut score = 0usize;
            match properties.v1_0.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU if self.allow_discrete_gpu => score += 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU if self.allow_integrated_gpu => score += 100,
                vk::PhysicalDeviceType::VIRTUAL_GPU if self.allow_virtual_gpu => score += 10,
                vk::PhysicalDeviceType::CPU if self.allow_cpu => score += 1,
                _ => continue,
            }

            // TODO: check for required features

            match &result {
                Some((_index, best_score)) if *best_score >= score => continue,
                _ => result = Some((index, score)),
            }
        }

        let (index, _) = result.ok_or(PhysicalDeviceSelectorError::NoPhysicalDeviceFound)?;
        let physical_device = self.physical_devices.swap_remove(index);

        // TODO: filter out unsupported features
        let supported_features = self.requested_features.keys().copied().collect();

        Ok(SelectedPhysicalDevice {
            physical_device,
            supported_features,
        })
    }
}

/// A physical device with a set of supported features.
pub struct SelectedPhysicalDevice {
    pub physical_device: gfx::PhysicalDevice,
    pub supported_features: FastHashSet<DeviceFeature>,
}

impl SelectedPhysicalDevice {
    /// Creates a logical device and a set of queues.
    pub fn create_logical_device<Q>(
        self,
        queues: Q,
    ) -> Result<(crate::device::Device, Q::Queues), CreateDeviceError<Q::Error>>
    where
        Q: QueuesQuery,
    {
        let features = self
            .supported_features
            .iter()
            .copied()
            .collect::<Box<[_]>>();
        self.physical_device.create_device(&features, queues)
    }
}

enum Necessity {
    Required,
    // Optional { score: usize },
}

/// Error that can occur when selecting a physical device.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PhysicalDeviceSelectorError {
    #[error("required device features are not supported: {0:?}")]
    RequiredFeaturesNotSupported(Vec<DeviceFeature>),
    #[error("no physical device found")]
    NoPhysicalDeviceFound,
}
