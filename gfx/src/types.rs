use vulkanalia::vk;

/// Buffer device address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceAddress(pub std::num::NonZeroU64);

impl DeviceAddress {
    pub fn new(address: u64) -> Option<Self> {
        std::num::NonZeroU64::new(address).map(Self)
    }
}

/// Pipeline value state.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum State<T> {
    Static(T),
    Dynamic,
}

impl<T> State<T> {
    #[inline]
    pub const fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic)
    }
}

/// Out of device memory error.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("a device memory allocation has failed")]
pub struct OutOfDeviceMemory;

impl OutOfDeviceMemory {
    pub(crate) fn on_creation(e: vk::ErrorCode) -> Self {
        match e {
            vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
            vk::ErrorCode::OUT_OF_DEVICE_MEMORY => Self,
            _ => crate::unexpected_vulkan_error(e),
        }
    }
}

/// Device lost error.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("the logical or physical device has been lost")]
pub struct DeviceLost;

/// Surface lost error.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("a surface is no longer available")]
pub struct SurfaceLost;
