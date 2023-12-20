use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use shared::util::WithDefer;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{KhrSurfaceExtension, KhrSwapchainExtension};
use winit::window::Window;

use crate::device::WeakDevice;
use crate::resources::{Image, ImageInfo, Samples, Semaphore};

pub struct Surface {
    window: Arc<Window>,
    handle: vk::SurfaceKHR,
    owner: WeakDevice,
    swapchain: Option<Swapchain>,
    unused_swapchains: VecDeque<Swapchain>,
    swapchain_support: SwapchainSupport,
    image_available: Semaphore,
}

impl Surface {
    pub(crate) fn new(
        handle: vk::SurfaceKHR,
        window: Arc<Window>,
        device: &crate::device::Device,
    ) -> Result<Self> {
        let instance = device.graphics().instance();
        let swapchain_support = SwapchainSupport::new(instance, device.physical(), handle)?;

        anyhow::ensure!(
            !swapchain_support.supported_families.is_empty(),
            "no queues with present capability found"
        );

        let image_available = device.create_semaphore()?;

        Ok(Surface {
            window,
            handle,
            owner: device.downgrade(),
            swapchain: None,
            unused_swapchains: VecDeque::new(),
            swapchain_support,
            image_available,
        })
    }

    pub fn swapchain_support(&self) -> &SwapchainSupport {
        &self.swapchain_support
    }

    pub fn update(&mut self) -> Result<()> {
        if let Some(swapchain) = &mut self.swapchain {
            let usage = swapchain.usage;
            let format = swapchain.format;
            let mode = swapchain.mode;
            self.configure_ext(usage, format, mode)
        } else {
            // TODO: configure with default best values instead?
            Ok(())
        }
    }

    pub fn configure(&mut self) -> Result<()> {
        let surface_format = self
            .swapchain_support
            .find_best_surface_format()
            .context("no suitable surface format found")?;

        let mode = self.swapchain_support.find_best_present_mode();

        self.configure_ext(
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            surface_format.format,
            mode,
        )
    }

    pub fn configure_ext(
        &mut self,
        usage: vk::ImageUsageFlags,
        format: vk::Format,
        mode: vk::PresentModeKHR,
    ) -> Result<()> {
        let device = self.owner.upgrade().context("device was already dropped")?;
        let instance = device.graphics().instance();
        let logical = device.logical();

        if self.unused_swapchains.len() > 16 {
            tracing::warn!("too many unused swapchains");
            device.wait_idle()?;
        }
        self.cleanup_unused_swapchains(&device);

        self.swapchain_support = SwapchainSupport::new(instance, device.physical(), self.handle)?;
        let capabilities = &self.swapchain_support.capabilities;
        anyhow::ensure!(
            capabilities.supported_usage_flags.contains(usage),
            "usage mode {usage:?} is not supported"
        );

        let surface_format = self
            .swapchain_support
            .surface_formats
            .iter()
            .find(|item| item.format == format)
            .with_context(|| format!("surface format {format:?} is not supported"))?;

        anyhow::ensure!(
            self.swapchain_support
                .present_modes
                .iter()
                .find(|item| **item == mode)
                .is_some(),
            "present mode {mode:?} is not supported"
        );

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count != 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let image_extent = self
            .swapchain_support
            .compute_swapchain_extent(&self.window);

        let composite_alpha = {
            let bits = capabilities.supported_composite_alpha.bits();
            if bits == 0 {
                vk::CompositeAlphaFlagsKHR::OPAQUE
            } else {
                vk::CompositeAlphaFlagsKHR::from_bits_truncate(1 << bits.trailing_zeros())
            }
        };

        let old_swapchain = if let Some(swapchain) = self.swapchain.take() {
            let handle = swapchain.handle;
            self.unused_swapchains.push_back(swapchain);
            handle
        } else {
            vk::SwapchainKHR::null()
        };

        let handle = {
            let info = vk::SwapchainCreateInfoKHR::builder()
                .surface(self.handle)
                .min_image_count(image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(image_extent)
                .image_array_layers(1)
                .image_usage(usage)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(composite_alpha)
                .present_mode(mode)
                .clipped(true)
                .old_swapchain(old_swapchain);

            unsafe { logical.create_swapchain_khr(&info, None) }?
        }
        .with_defer(|swapchain| unsafe { logical.destroy_swapchain_khr(swapchain, None) });

        let images = unsafe { logical.get_swapchain_images_khr(*handle) }?;
        let images = images
            .into_iter()
            .map(|handle| {
                let acquire = device.create_semaphore()?;
                let release = device.create_semaphore()?;
                let info = ImageInfo {
                    extent: image_extent.into(),
                    format,
                    mip_levels: 1,
                    samples: Samples::_1,
                    layers: 1,
                    usage,
                };
                let id = IMAGE_ID.fetch_add(1, Ordering::Relaxed).try_into().unwrap();
                let image = Image::new_surface(handle, info, device.downgrade(), id);
                Ok::<_, anyhow::Error>(SwapchainImageState {
                    image,
                    acquire,
                    release,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let handle = handle.disarm();

        self.swapchain = Some(Swapchain {
            handle,
            format,
            usage,
            mode,
            images,
            optimal: true,
            acquired_count: 0,
        });

        tracing::debug!(swapchain = ?handle, "created swapchain");

        Ok(())
    }

    pub fn aquire_image(&mut self) -> Result<SurfaceImage<'_>> {
        let device = self.owner.upgrade().context("device was already dropped")?;
        self.cleanup_unused_swapchains(&device);

        let index = loop {
            let swapchain = self
                .swapchain
                .as_mut()
                .context("swapchain not configured")?;

            let available_images =
                swapchain.images.len() as u32 - self.swapchain_support.capabilities.min_image_count;
            anyhow::ensure!(
                swapchain.acquired_count <= available_images,
                "too many acquired images"
            );

            let res = unsafe {
                device.logical().acquire_next_image_khr(
                    swapchain.handle,
                    u64::MAX,
                    self.image_available.handle(),
                    vk::Fence::null(),
                )
            };

            match res {
                Ok((index, code)) => {
                    if code == vk::SuccessCode::SUBOPTIMAL_KHR {
                        swapchain.optimal = false;
                    }
                    break index;
                }
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                    let usage = swapchain.usage;
                    let format = swapchain.format;
                    let mode = swapchain.mode;
                    self.configure_ext(usage, format, mode)?;
                    continue;
                }
                Err(e) => anyhow::bail!("failed to acquire next swapchain image: {e}"),
            }
        };

        let swapchain = self.swapchain.as_mut().unwrap();

        let image_state = &mut swapchain.images[index as usize];
        std::mem::swap(&mut image_state.acquire, &mut self.image_available);
        swapchain.acquired_count += 1;

        Ok(SurfaceImage {
            handle: swapchain.handle,
            supported_families: &self.swapchain_support.supported_families,
            image: &image_state.image,
            wait: &mut image_state.acquire,
            signal: &mut image_state.release,
            optimal: swapchain.optimal,
            used: false,
        })
    }

    fn cleanup_unused_swapchains(&mut self, device: &crate::device::Device) {
        let logical = device.logical();

        // For each unused swapchain starting from the oldest
        while let Some(mut swapchain) = self.unused_swapchains.pop_front() {
            // For each remaining image
            while let Some(mut state) = swapchain.images.pop() {
                // Dispose it only if it is not used anywhere else
                if let Err(image) = state.image.try_dispose_as_surface() {
                    // Revert otherwise
                    state.image = image;
                    swapchain.images.push(state);
                    self.unused_swapchains.push_front(swapchain);
                    // And stop processing immediately
                    return;
                }
            }

            // Swapchain with no shared images can be safely destroyed
            unsafe { logical.destroy_swapchain_khr(swapchain.handle, None) };
        }
    }
}

pub struct SurfaceImage<'a> {
    handle: vk::SwapchainKHR,
    supported_families: &'a [bool],
    image: &'a Image,
    wait: &'a mut Semaphore,
    signal: &'a mut Semaphore,
    optimal: bool,
    used: bool,
}

impl<'a> SurfaceImage<'a> {
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub fn supported_families(&self) -> &'a [bool] {
        self.supported_families
    }

    pub fn image(&self) -> &'a Image {
        self.image
    }

    pub fn wait_signal(&mut self) -> [&mut Semaphore; 2] {
        [&mut *self.wait, &mut *self.signal]
    }

    pub fn is_optimal(&self) -> bool {
        self.optimal
    }

    pub fn consume(mut self) {
        self.used = true;
    }
}

impl Drop for SurfaceImage<'_> {
    fn drop(&mut self) {
        if self.used && !std::thread::panicking() {
            tracing::error!("surface image was not presented")
        }
    }
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    mode: vk::PresentModeKHR,
    images: Vec<SwapchainImageState>,
    acquired_count: u32,
    optimal: bool,
}

struct SwapchainImageState {
    image: Image,
    acquire: Semaphore,
    release: Semaphore,
}

pub struct SwapchainSupport {
    pub supported_families: Box<[bool]>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let queue_family_count = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .len()
        };

        let supported_families = (0..queue_family_count as u32)
            .map(|family_idx| unsafe {
                instance.get_physical_device_surface_support_khr(
                    physical_device,
                    family_idx,
                    surface,
                )
            })
            .collect::<Result<Box<[_]>, _>>()?;

        let capabilities = unsafe {
            instance.get_physical_device_surface_capabilities_khr(physical_device, surface)
        }?;
        let surface_formats =
            unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface) }?;
        let present_modes = unsafe {
            instance.get_physical_device_surface_present_modes_khr(physical_device, surface)
        }?;

        Ok(Self {
            supported_families,
            capabilities,
            surface_formats,
            present_modes,
        })
    }

    pub fn find_best_surface_format(&self) -> Option<vk::SurfaceFormatKHR> {
        const TARGET: vk::Format = vk::Format::B8G8R8A8_SRGB;
        const COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;

        let mut alternative_target = None;
        for &item in &self.surface_formats {
            if item.format == TARGET && item.color_space == COLOR_SPACE {
                return Some(item);
            } else if alternative_target.is_none() && item.color_space == COLOR_SPACE {
                alternative_target = Some(item);
            }
        }

        alternative_target.or(self.surface_formats.first().copied())
    }

    pub fn find_best_present_mode(&self) -> vk::PresentModeKHR {
        const TARGET: vk::PresentModeKHR = vk::PresentModeKHR::MAILBOX;
        const FALLBACK: vk::PresentModeKHR = vk::PresentModeKHR::FIFO;

        self.present_modes
            .iter()
            .copied()
            .find(|p| *p == TARGET)
            .unwrap_or(FALLBACK)
    }

    pub fn compute_swapchain_extent(&self, window: &Window) -> vk::Extent2D {
        let capabilities = &self.capabilities;

        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        let size = window.inner_size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.height,
            ))
            .build()
    }
}

const IMAGE_ID: AtomicU64 = AtomicU64::new(1);
