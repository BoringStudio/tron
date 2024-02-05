use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::{c_void, CStr, CString};
use std::sync::Mutex;

use once_cell::sync::OnceCell;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::ExtDebugUtilsExtension as _;
use vulkanalia::Instance;

use crate::physical::{PhysicalDevice, PhysicalDeviceSelector};
use crate::types::OutOfDeviceMemory;

/// Graphics instance configuration.
#[derive(Debug, Clone)]
pub struct InstanceConfig {
    pub app_name: Cow<'static, str>,
    pub app_version: (u32, u32, u32),
    pub validation_layer_enabled: bool,
}

/// Graphics instance.
pub struct Graphics {
    instance: Instance,
    api_version: u32,
    config: InstanceConfig,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    _entry: Entry,
}

impl Graphics {
    /// Sets the initial [`InstanceConfig`] to be used when initializing.
    pub fn set_init_config(config: InstanceConfig) {
        if GRAPHICS.get().is_some() {
            tracing::warn!("updating instance config after graphics initialization has no effect");
        }
        *INIT_CONFIG.lock().unwrap() = config;
    }

    /// Returns an initialized Vulkan instance wrapper or
    /// creates it using the current [`InstanceConfig`].
    ///
    /// See [`set_init_config`].
    ///
    /// [`set_init_config`]: [`Graphics::set_init_config`]
    pub fn get_or_init() -> Result<&'static Self, InitGraphicsError> {
        GRAPHICS.get_or_try_init(|| unsafe { Self::new() })
    }

    /// Returns an initialized Vulkan instance wrapper.
    ///
    /// # Safety
    ///
    /// The following must be true:
    /// - [`Graphics`] must have been previously initialized
    pub unsafe fn get_unchecked() -> &'static Self {
        GRAPHICS.get_unchecked()
    }

    unsafe fn new() -> Result<Self, InitGraphicsError> {
        let config = INIT_CONFIG.lock().unwrap().clone();

        // Init API entry
        let loader = LibloadingLoader::new(LIBRARY)
            .map_err(|e| InitGraphicsError::EntryLoadFailed(Box::new(e)))?;
        let entry = Entry::new(loader).map_err(|e| {
            InitGraphicsError::EntryLoadFailed(Box::new(std::io::Error::other(e.to_string())))
        })?;

        // Prepare basic app info
        let api_version = match entry.version() {
            Ok(version) => version.into(),
            Err(e) => crate::unexpected_vulkan_error(e),
        };
        let app_name =
            CString::new(config.app_name.as_ref()).expect("app name must not contain null");
        let app_version = vk::make_version(
            config.app_version.0,
            config.app_version.1,
            config.app_version.2,
        );
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_bytes())
            .application_version(app_version)
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(0, 0, 1))
            .api_version(api_version);

        // Get available layers and extensions
        fn map_enumerate_err(e: vk::ErrorCode) -> InitGraphicsError {
            match e {
                vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => {
                    InitGraphicsError::OutOfDeviceMemory(OutOfDeviceMemory)
                }
                vk::ErrorCode::LAYER_NOT_PRESENT => InitGraphicsError::LayerNotLoaded,
                _ => crate::unexpected_vulkan_error(e),
            }
        }

        let available_layers = entry
            .enumerate_instance_layer_properties()
            .map_err(map_enumerate_err)?
            .into_iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();
        let available_extensions = entry
            .enumerate_instance_extension_properties(None)
            .map_err(map_enumerate_err)?
            .into_iter()
            .map(|item| item.extension_name)
            .collect::<HashSet<_>>();

        let mut layers = Vec::new();
        let mut push_layer = |layer: &'static vk::ExtensionName| {
            let available = available_layers.contains(layer);
            if available {
                layers.push(layer.as_ptr());
            }
            available
        };

        let mut extensions = Vec::new();
        let mut push_ext = |ext: &'static vk::Extension| -> bool {
            let available = available_extensions.contains(&ext.name);
            if available {
                extensions.push(ext.name.as_ptr());
            }
            available
        };

        // Add validation layer extensions
        let validation_enabled = config.validation_layer_enabled && {
            static VALIDATION_LAYER: vk::ExtensionName =
                vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
            static ALT_VALIDATION_LAYER: vk::ExtensionName =
                vk::ExtensionName::from_bytes(b"VK_LAYER_LUNARG_standard_validation");

            if push_layer(&VALIDATION_LAYER) || push_layer(&ALT_VALIDATION_LAYER) {
                push_ext(&vk::EXT_DEBUG_UTILS_EXTENSION)
            } else {
                tracing::warn!("Vulkan validation layers are not available");
                false
            }
        };

        // Add required extensions for creating windows
        #[cfg(target_os = "macos")]
        let flags = if Self::requires_portability(api_version) {
            push_ext(&vk::KHR_PORTABILITY_ENUMERATION_EXTENSION);
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };
        #[cfg(not(target_os = "macos"))]
        let flags = vk::InstanceCreateFlags::empty();

        push_ext(&vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION);

        let supports_surface = push_ext(&vk::KHR_SURFACE_EXTENSION);
        if !supports_surface {
            // Running on calculator?
            panic!("Vulkan surface extension support is mandatory");
        }

        #[cfg(any(
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        {
            push_ext(&vk::KHR_XLIB_SURFACE_EXTENSION);
            push_ext(&vk::KHR_XCB_SURFACE_EXTENSION);
            push_ext(&vk::KHR_WAYLAND_SURFACE_EXTENSION);
        }

        #[cfg(target_os = "windows")]
        {
            push_ext(&vk::KHR_WIN32_SURFACE_EXTENSION);
        }

        #[cfg(target_os = "macos")]
        {
            push_ext(&vk::EXT_METAL_SURFACE_EXTENSION);
        }

        let mut instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers)
            .flags(flags);

        let mut debug_info = make_debug_callback_info();
        if validation_enabled {
            instance_info = instance_info.push_next(&mut debug_info);
        }

        let instance = entry
            .create_instance(&instance_info, None)
            .map_err(|e| match e {
                vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => {
                    InitGraphicsError::OutOfDeviceMemory(OutOfDeviceMemory)
                }
                vk::ErrorCode::INITIALIZATION_FAILED => InitGraphicsError::InitializationFailed,
                // NOTE: no unsupported layers are requested, but this error can still occur
                // if it fails to load the validation layer
                vk::ErrorCode::LAYER_NOT_PRESENT => InitGraphicsError::LayerNotLoaded,
                vk::ErrorCode::INCOMPATIBLE_DRIVER => InitGraphicsError::IncompatibleDriver,
                // NOTE: `EXTENSION_NOT_PRESENT` is also unexpected because we check for
                // extension support before creating the instance
                _ => crate::unexpected_vulkan_error(e),
            })?;

        let debug_utils_messenger = if validation_enabled {
            let debug_info = make_debug_callback_info();
            match instance.create_debug_utils_messenger_ext(&debug_info, None) {
                Ok(handle) => handle,
                Err(e) => match e {
                    vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
                    _ => crate::unexpected_vulkan_error(e),
                },
            }
        } else {
            vk::DebugUtilsMessengerEXT::null()
        };

        Ok(Self {
            instance,
            api_version,
            config,
            debug_utils_messenger,
            _entry: entry,
        })
    }

    /// Returns the [`InstanceConfig`] used to initialize the instance.
    pub fn config(&self) -> &InstanceConfig {
        &self.config
    }

    /// Returns the [`PhysicalDevice`]s available on the system.
    pub fn get_physical_devices(&self) -> Result<PhysicalDeviceSelector, OutOfDeviceMemory> {
        let devices =
            unsafe { self.instance.enumerate_physical_devices() }.map_err(|e| match e {
                vk::ErrorCode::OUT_OF_HOST_MEMORY => crate::out_of_host_memory(),
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => OutOfDeviceMemory,
                _ => crate::unexpected_vulkan_error(e),
            })?;

        Ok(PhysicalDeviceSelector::new(
            devices
                .into_iter()
                .map(|handle| unsafe { PhysicalDevice::new(handle) })
                .collect(),
        ))
    }

    /// Returns the underlying Vulkan instance.
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Returns the Vulkan API version.
    pub fn api_version(&self) -> u32 {
        self.api_version
    }

    /// Returns `true` if the Vulkan API version is at least 1.1.
    pub fn vk1_1(&self) -> bool {
        vk::version_major(self.api_version) >= 1 && vk::version_minor(self.api_version) >= 1
    }

    /// Returns `true` if the Vulkan API version is at least 1.2.
    pub fn vk1_2(&self) -> bool {
        vk::version_major(self.api_version) >= 1 && vk::version_minor(self.api_version) >= 2
    }

    /// Returns `true` if the Vulkan API version is at least 1.3.
    pub fn vk1_3(&self) -> bool {
        vk::version_major(self.api_version) >= 1 && vk::version_minor(self.api_version) >= 3
    }

    #[cfg(target_os = "macos")]
    pub(crate) const fn requires_portability(api_version: u32) -> bool {
        const PORTABILITY_MACOS_VERSION: u32 = vk::make_version(1, 3, 216);
        api_version >= PORTABILITY_MACOS_VERSION
    }
}

impl Drop for Graphics {
    fn drop(&mut self) {
        unsafe {
            if !self.debug_utils_messenger.is_null() {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.debug_utils_messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

fn make_debug_callback_info() -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback))
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*data).message).to_string_lossy();

    // TODO: optimize
    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        tracing::error!(target: "validation", ?ty, "{message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        tracing::warn!(target: "validation", ?ty, "{message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        tracing::debug!(target: "validation", ?ty, "{message}");
    } else {
        tracing::trace!(target: "validation", ?ty, "{message}");
    };

    vk::FALSE
}

static GRAPHICS: OnceCell<Graphics> = OnceCell::new();
static INIT_CONFIG: Mutex<InstanceConfig> = Mutex::new(InstanceConfig {
    app_name: Cow::Borrowed("app"),
    app_version: (0, 0, 1),
    validation_layer_enabled: true,
});

/// An error returned when initializing the graphics instance fails.
#[derive(Debug, thiserror::Error)]
pub enum InitGraphicsError {
    #[error(transparent)]
    OutOfDeviceMemory(#[from] OutOfDeviceMemory),
    #[error("failed to load Vulkan entry point")]
    EntryLoadFailed(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("instance initialization could not be completed for implementation-specific reasons")]
    InitializationFailed,
    #[error("some requested layers could not be loaded")]
    LayerNotLoaded,
    #[error(
        "the requested version of Vulkan is not supported by the driver or \
        is otherwise incompatible for implementation-specific reasons"
    )]
    IncompatibleDriver,
}
