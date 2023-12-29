use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::{c_void, CStr, CString};
use std::sync::Mutex;

use anyhow::Result;
use once_cell::sync::OnceCell;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::ExtDebugUtilsExtension as _;
use vulkanalia::Instance;
use winit::raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
};

use crate::PhysicalDevice;

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
    pub fn get_or_init() -> Result<&'static Self> {
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

    unsafe fn new() -> Result<Self> {
        let config = INIT_CONFIG.lock().unwrap().clone();

        // Init API entry
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(anyhow::Error::msg)?;

        // Prepare basic app info
        let api_version = entry.version()?.into();
        let app_name = CString::new(config.app_name.as_ref())?;
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
        let available_layers = entry
            .enumerate_instance_layer_properties()?
            .into_iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();
        let available_extensions = entry
            .enumerate_instance_extension_properties(None)?
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
        push_ext(&vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION);
        anyhow::ensure!(
            push_ext(&vk::KHR_SURFACE_EXTENSION),
            "Vulkan surface extension support is mandatory"
        );

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

        // TODO: macos

        let mut instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);

        let mut debug_info = make_debug_callback_info();
        if validation_enabled {
            instance_info = instance_info.push_next(&mut debug_info);
        }

        let instance = entry.create_instance(&instance_info, None)?;

        let debug_utils_messenger = if validation_enabled {
            let debug_info = make_debug_callback_info();
            instance.create_debug_utils_messenger_ext(&debug_info, None)?
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
    pub fn get_physical_devices(&self) -> Result<Vec<PhysicalDevice>> {
        let devices = unsafe { self.instance.enumerate_physical_devices() }?;

        Ok(devices
            .into_iter()
            .map(|handle| unsafe { PhysicalDevice::new(handle) })
            .collect())
    }

    pub(crate) fn create_raw_surface<W>(&self, window: &W) -> Result<vk::SurfaceKHR>
    where
        W: HasDisplayHandle + HasWindowHandle,
    {
        let require_extension = |ext: &vk::Extension| -> Result<()> {
            anyhow::ensure!(
                self.instance.extensions().contains(&ext.name),
                "`{}` is not supported",
                ext.name
            );
            Ok(())
        };

        match (
            window.display_handle().map(|handle| handle.as_raw())?,
            window.window_handle().map(|handle| handle.as_raw())?,
        ) {
            #[cfg(target_os = "windows")]
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
                use vk::KhrWin32SurfaceExtension;

                require_extension(&vk::KHR_WIN32_SURFACE_EXTENSION)?;

                let hinstance_ptr = window
                    .hinstance
                    .map(|hinstance| hinstance.get() as vk::HINSTANCE)
                    .unwrap_or(std::ptr::null_mut());
                let hwnd_ptr = window.hwnd.get() as vk::HWND;

                let info = vk::Win32SurfaceCreateInfoKHR::builder()
                    .hinstance(hinstance_ptr)
                    .hwnd(hwnd_ptr);

                unsafe { self.instance.create_win32_surface_khr(&info, None) }
            }
            #[cfg(any(
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "netbsd",
                target_os = "openbsd"
            ))]
            (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
                use vk::KhrXcbSurfaceExtension;

                require_extension(&vk::KHR_XCB_SURFACE_EXTENSION)?;

                let connection_ptr = display
                    .connection
                    .map(|connection| connection.as_ptr())
                    .unwrap_or(std::ptr::null_mut());

                let info = vk::XcbSurfaceCreateInfoKHR::builder()
                    .window(window.window.get())
                    .connection(connection_ptr);

                unsafe { self.instance.create_xcb_surface_khr(&info, None) }
            }
            #[cfg(any(
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "netbsd",
                target_os = "openbsd"
            ))]
            (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
                use vk::KhrXlibSurfaceExtension;

                require_extension(&vk::KHR_XLIB_SURFACE_EXTENSION)?;

                let display_ptr = display
                    .display
                    .map(|display| display.as_ptr())
                    .unwrap_or(std::ptr::null_mut());

                let info = vk::XlibSurfaceCreateInfoKHR {
                    dpy: display_ptr.cast(),
                    window: window.window,
                    ..Default::default()
                };

                unsafe { self.instance.create_xlib_surface_khr(&info, None) }
            }
            #[cfg(any(
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "netbsd",
                target_os = "openbsd"
            ))]
            (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
                use vk::KhrWaylandSurfaceExtension;

                require_extension(&vk::KHR_WAYLAND_SURFACE_EXTENSION)?;

                let info = vk::WaylandSurfaceCreateInfoKHR::builder()
                    .display(display.display.as_ptr())
                    .surface(window.surface.as_ptr());

                unsafe { self.instance.create_wayland_surface_khr(&info, None) }
            }
            _ => anyhow::bail!("unsupported window and display kind combination"),
        }
        .map_err(Into::into)
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
