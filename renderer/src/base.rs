use std::collections::HashSet;
use std::ffi::{c_void, CStr, CString};

use anyhow::{Context, Result};
use shared::util::WithDefer;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension};
use vulkanalia::window as vk_window;
use winit::window::Window;

use super::RendererConfig;

pub struct RendererBase {
    device: Device,
    queues: Queues,
    physical_device: PhysicalDevice,
    surface: vk::SurfaceKHR,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    instance: Instance,
    loader_entry: Entry,
}

impl RendererBase {
    pub unsafe fn new(window: &Window, config: RendererConfig) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let loader_entry = Entry::new(loader).map_err(anyhow::Error::msg)?;
        let instance = create_instance(window, &loader_entry, &config)?;

        let debug_utils_messenger = if config.validation_layer_enabled {
            let debug_info = make_debug_callback_info();
            instance.create_debug_utils_messenger_ext(&debug_info, None)?
        } else {
            vk::DebugUtilsMessengerEXT::null()
        }
        .with_defer(|messenger| {
            if !messenger.is_null() {
                instance.destroy_debug_utils_messenger_ext(messenger, None);
            }
        });

        let surface = vk_window::create_surface(&instance, window, window)?
            .with_defer(|surface| instance.destroy_surface_khr(surface, None));

        let physical_device = find_physical_device(&instance, *surface)?;
        let (device, queues) = physical_device.create_logical_device(&instance, &config)?;

        Ok(Self {
            device,
            queues,
            physical_device,
            surface: surface.disarm(),
            debug_utils_messenger: debug_utils_messenger.disarm(),
            instance,
            loader_entry,
        })
    }

    pub unsafe fn compute_swapchain_support(&self) -> Result<SwapchainSupport> {
        SwapchainSupport::new(&self.instance, self.physical_device.handle, self.surface)
    }

    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[inline]
    pub fn queues(&self) -> &Queues {
        &self.queues
    }

    #[inline]
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }

    #[inline]
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    #[inline]
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
}

impl Drop for RendererBase {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);

            if !self.surface.is_null() {
                self.instance.destroy_surface_khr(self.surface, None);
            }

            if !self.debug_utils_messenger.is_null() {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.debug_utils_messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    config: &RendererConfig,
) -> Result<Instance> {
    let app_name = CString::new(config.app_name.as_str())?;
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
        .api_version(vk::make_version(1, 0, 0));

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .into_iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    let mut layers = Vec::new();
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if config.validation_layer_enabled {
        anyhow::ensure!(
            available_layers.contains(&VALIDATION_LAYER),
            "Requested vulkan validation layer not found"
        );
        layers.push(VALIDATION_LAYER.as_ptr());
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let mut instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&extensions)
        .enabled_layer_names(&layers);

    let mut debug_info = make_debug_callback_info();
    if config.validation_layer_enabled {
        instance_info = instance_info.push_next(&mut debug_info);
    }

    entry
        .create_instance(&instance_info, None)
        .map_err(From::from)
}

unsafe fn find_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
) -> Result<PhysicalDevice> {
    let mut result = None;
    for device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(device);

        match PhysicalDevice::new(instance, device, &properties, surface) {
            Ok((score, device)) => {
                if !matches!(&result, Some((prev_score, _)) if *prev_score >= score) {
                    result = Some((score, device));
                }
            }
            Err(e) => {
                tracing::warn!(
                    device = %properties.device_name,
                    device_type = ?properties.device_type,
                    "skipping physical device: {e:?}"
                )
            }
        }
    }

    let (_, result) = result.context("No suitable physical device found")?;
    let properties = instance.get_physical_device_properties(result.handle);
    tracing::info!(
        device = %properties.device_name,
        device_type = ?properties.device_type,
        present_queue_family_idx = result.present_queue_family_idx,
        graphics_queue_family_idx = result.graphics_queue_family_idx,
        compute_queue_family_idx = result.compute_queue_family_idx,
        "found a suitable physical device"
    );

    Ok(result)
}

pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub present_queue_family_idx: u32,
    pub graphics_queue_family_idx: u32,
    pub compute_queue_family_idx: u32,
}

impl PhysicalDevice {
    unsafe fn new(
        instance: &Instance,
        device: vk::PhysicalDevice,
        properties: &vk::PhysicalDeviceProperties,
        surface: vk::SurfaceKHR,
    ) -> Result<(usize, Self)> {
        let queue_family_properties = instance.get_physical_device_queue_family_properties(device);

        let (present_queue_family_idx, graphics_queue_family_idx, compute_queue_family_idx) = {
            let mut present = None;
            let mut graphics = None;
            let mut compute = None;

            for (idx, queue) in queue_family_properties.into_iter().enumerate() {
                let idx = idx as u32;

                if present.is_none()
                    && instance.get_physical_device_surface_support_khr(device, idx, surface)?
                {
                    present = Some(idx);
                }

                if graphics.is_none() && queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    graphics = Some(idx);
                }

                if compute.is_none() && queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    compute = Some(idx);
                }
            }

            (
                present.context("Present queue family not found")?,
                graphics.context("Graphics queue family not found")?,
                compute.context("Compute queue family not found")?,
            )
        };

        let extensions = instance
            .enumerate_device_extension_properties(device, None)?
            .into_iter()
            .map(|p| p.extension_name)
            .collect::<HashSet<_>>();
        anyhow::ensure!(
            DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)),
            "Missing required device extensions"
        );

        let mut score = 0;
        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            score += 1000;
        }

        let result = PhysicalDevice {
            handle: device,
            present_queue_family_idx,
            graphics_queue_family_idx,
            compute_queue_family_idx,
        };
        Ok((score, result))
    }

    unsafe fn create_logical_device(
        &self,
        instance: &Instance,
        config: &RendererConfig,
    ) -> Result<(Device, Queues)> {
        let queue_priorities = &[1.0];

        let queue_indices = HashSet::from([
            self.present_queue_family_idx,
            self.graphics_queue_family_idx,
            self.compute_queue_family_idx,
        ]);
        let queue_infos = queue_indices
            .into_iter()
            .map(|idx| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(idx)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();

        let mut layers = Vec::new();
        if config.validation_layer_enabled {
            layers.push(VALIDATION_LAYER.as_ptr());
        }

        // TODO: extend
        let extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let features = vk::PhysicalDeviceFeatures::builder();

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = instance.create_device(self.handle, &device_info, None)?;
        let queues = Queues {
            present_queue: device.get_device_queue(self.present_queue_family_idx, 0),
            graphics_queue: device.get_device_queue(self.graphics_queue_family_idx, 0),
            compute_queue: device.get_device_queue(self.compute_queue_family_idx, 0),
        };

        Ok((device, queues))
    }
}

pub struct Queues {
    pub present_queue: vk::Queue,
    pub graphics_queue: vk::Queue,
    pub compute_queue: vk::Queue,
}

pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
}

impl SwapchainSupport {
    unsafe fn new(
        instance: &Instance,
        device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        fn find_best_surface_format(
            formats: &[vk::SurfaceFormatKHR],
        ) -> Result<vk::SurfaceFormatKHR> {
            const TARGET: vk::Format = vk::Format::B8G8R8A8_SRGB;
            const COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;

            let mut alternative_target = None;
            for item in formats {
                if item.format == TARGET && item.color_space == COLOR_SPACE {
                    return Ok(*item);
                } else if alternative_target.is_none() && item.color_space == COLOR_SPACE {
                    alternative_target = Some(item);
                }
            }

            alternative_target
                .or(formats.first())
                .copied()
                .context("Suitable surface format not found")
        }

        fn find_best_present_mode(
            present_modes: &[vk::PresentModeKHR],
        ) -> Result<vk::PresentModeKHR> {
            const TARGET: vk::PresentModeKHR = vk::PresentModeKHR::MAILBOX;
            const FALLBACK: vk::PresentModeKHR = vk::PresentModeKHR::FIFO;

            anyhow::ensure!(!present_modes.is_empty(), "Suitable present mode not found");
            Ok(present_modes
                .iter()
                .cloned()
                .find(|p| *p == TARGET)
                .unwrap_or(FALLBACK))
        }

        let capabilities =
            instance.get_physical_device_surface_capabilities_khr(device, surface)?;
        let surface_formats = instance.get_physical_device_surface_formats_khr(device, surface)?;
        let present_modes =
            instance.get_physical_device_surface_present_modes_khr(device, surface)?;

        Ok(Self {
            capabilities,
            surface_format: find_best_surface_format(&surface_formats)?,
            present_mode: find_best_present_mode(&present_modes)?,
        })
    }
}

fn make_debug_callback_info() -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback))
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).message).to_string_lossy() };

    // TODO: optimize
    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        tracing::error!(?ty, "{message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        tracing::warn!(?ty, "{message}");
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        tracing::debug!(?ty, "{message}");
    } else {
        tracing::trace!(?ty, "{message}");
    };

    vk::FALSE
}

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
