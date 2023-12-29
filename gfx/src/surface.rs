use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use shared::util::WithDefer;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{KhrSurfaceExtension, KhrSwapchainExtension};
use winit::window::Window;

use crate::device::WeakDevice;
use crate::resources::{Format, Image, ImageInfo, ImageUsageFlags, Samples, Semaphore};
use crate::util::{FromGfx, ToVk, TryFromVk};

/// Presentation mode supported for a surface.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PresentMode {
    /// Presentation engine does not wait for a vertical blanking period to update the current image.
    ///
    /// No internal queuing of presentation requests is needed, as the requests are applied immediately.
    Immediate,

    /// Presentation engine waits for a vertical blanking period to update the current image.
    ///
    /// An internal single-entry queue is used to hold pending presentation requests.
    /// If the queue is full when a new presentation request is received, the new request
    /// replaces the existing entry, and any images associated with the prior entry become
    /// available for reuse by the application. One request is removed from the queue and
    /// processed during each vertical blanking period in which the queue is non-empty.
    Mailbox,

    /// Presentation engine waits for a vertical blanking period to update the current image.
    ///
    /// An internal queue is used to hold pending presentation requests. New requests are
    /// appended to the end of the queue, and one request is removed from the beginning of
    /// the queue and processed during each vertical blanking period in which the queue is
    /// non-empty. This is the only value of [`PresentMode`] that is required to be supported.
    Fifo,

    /// Presentation engine generally waits for a vertical blanking period to update the current image.
    /// If a vertical blanking period has already passed since the last update of the current image
    /// then the presentation engine does not wait for another vertical blanking period for the update.
    ///
    /// This mode is useful for reducing visual stutter with an application that will mostly present
    /// a new image before the next vertical blanking period, but may occasionally be late, and
    /// present a new image just after the next vertical blanking period. An internal queue is used
    /// to hold pending presentation requests. New requests are appended to the end of the queue,
    /// and one request is removed from the beginning of the queue and processed during or after
    /// each vertical blanking period in which the queue is non-empty.
    FifoRelaxed,
}

impl TryFromVk<vk::PresentModeKHR> for PresentMode {
    fn try_from_vk(mode: vk::PresentModeKHR) -> Option<Self> {
        match mode {
            vk::PresentModeKHR::IMMEDIATE => Some(Self::Immediate),
            vk::PresentModeKHR::MAILBOX => Some(Self::Mailbox),
            vk::PresentModeKHR::FIFO => Some(Self::Fifo),
            vk::PresentModeKHR::FIFO_RELAXED => Some(Self::FifoRelaxed),
            _ => None,
        }
    }
}

impl FromGfx<PresentMode> for vk::PresentModeKHR {
    fn from_gfx(mode: PresentMode) -> Self {
        match mode {
            PresentMode::Immediate => Self::IMMEDIATE,
            PresentMode::Mailbox => Self::MAILBOX,
            PresentMode::Fifo => Self::FIFO,
            PresentMode::FifoRelaxed => Self::FIFO_RELAXED,
        }
    }
}

/// Wrapper around a surface object.
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

    /// Returns swapchain properties.
    pub fn swapchain_support(&self) -> &SwapchainSupport {
        &self.swapchain_support
    }

    /// Recreates the swapchain with the last parameters.
    ///
    /// NOTE: doesn't initialize the swapchain if it wasn't initialized before.
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

    /// Configures the swapchain with the best parameters.
    pub fn configure(&mut self) -> Result<()> {
        let format = self
            .swapchain_support
            .find_best_surface_format()
            .context("no suitable surface format found")?;

        let mode = self.swapchain_support.find_best_present_mode();

        self.configure_ext(ImageUsageFlags::COLOR_ATTACHMENT, format, mode)
    }

    /// Configures the swapchain with the specified parameters.
    pub fn configure_ext(
        &mut self,
        usage: ImageUsageFlags,
        format: Format,
        mode: PresentMode,
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
            capabilities.supported_usage_flags.contains(usage.to_vk()),
            "usage mode {usage:?} is not supported"
        );

        let surface_format = self
            .swapchain_support
            .surface_formats
            .iter()
            .find(|item| Format::from_vk(item.format) == Some(format))
            .with_context(|| format!("surface format {format:?} is not supported"))?;

        anyhow::ensure!(
            self.swapchain_support
                .present_modes
                .iter()
                .copied()
                .filter_map(PresentMode::try_from_vk)
                .any(|item| item == mode),
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
                .image_usage(usage.to_vk())
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(composite_alpha)
                .present_mode(mode.to_vk())
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
                    array_layers: 1,
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
        let image_count = images.len();

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

        tracing::debug!(
            swapchain = ?handle,
            image_count,
            format = ?surface_format.format,
            color_space = ?surface_format.color_space,
            ?mode,
            "created swapchain",
        );

        Ok(())
    }

    /// Acquires the next image from the swapchain.
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
                "too many acquired images: acquired {}, available {}",
                swapchain.acquired_count,
                available_images
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

        let total_image_count = swapchain.images.len();
        let image_state = &mut swapchain.images[index as usize];
        std::mem::swap(&mut image_state.acquire, &mut self.image_available);
        swapchain.acquired_count += 1;

        Ok(SurfaceImage {
            handle: swapchain.handle,
            supported_families: &self.swapchain_support.supported_families,
            total_image_count,
            image: &image_state.image,
            index,
            acquired_count: &mut swapchain.acquired_count,
            wait: &mut image_state.acquire,
            signal: &mut image_state.release,
            optimal: swapchain.optimal,
            used: false,
        })
    }

    fn cleanup_unused_swapchains(&mut self, device: &crate::device::Device) {
        let logical = device.logical();

        // For each unused swapchain starting from the oldest
        while let Some(swapchain) = self.unused_swapchains.front_mut() {
            // For each remaining image
            while let Some(mut state) = swapchain.images.pop() {
                // Dispose it only if it is not used anywhere else
                if let Err(image) = state.image.try_dispose_as_surface() {
                    // Revert otherwise
                    state.image = image;
                    swapchain.images.push(state);
                    return;
                }
            }

            // Swapchain with no shared images can be safely destroyed
            unsafe { logical.destroy_swapchain_khr(swapchain.handle, None) };
            self.unused_swapchains.pop_front();
        }
    }
}

/// Aquired image from a swapchain.
pub struct SurfaceImage<'a> {
    handle: vk::SwapchainKHR,
    supported_families: &'a [bool],
    total_image_count: usize,
    image: &'a Image,
    index: u32,
    acquired_count: &'a mut u32,
    wait: &'a mut Semaphore,
    signal: &'a mut Semaphore,
    optimal: bool,
    used: bool,
}

impl<'a> SurfaceImage<'a> {
    /// Returns the total number of images in the swapchain.
    pub fn total_image_count(&self) -> usize {
        self.total_image_count
    }

    /// Returns the swapchan image.
    pub fn image(&self) -> &'a Image {
        self.image
    }

    /// Returns the index of the image in the swapchain.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Returns the semaphore that should be waited on before using the image,
    /// and the semaphore that should be signaled after using the image.
    pub fn wait_signal(&mut self) -> [&mut Semaphore; 2] {
        [&mut *self.wait, &mut *self.signal]
    }

    /// Returns `true` if the image is optimal for presentation.
    pub fn is_optimal(&self) -> bool {
        self.optimal
    }

    pub(crate) fn swapchain_handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub(crate) fn supported_families(&self) -> &'a [bool] {
        self.supported_families
    }

    pub(crate) fn consume(mut self) {
        self.used = true;
        *self.acquired_count -= 1;
    }
}

impl Drop for SurfaceImage<'_> {
    fn drop(&mut self) {
        if !self.used && !std::thread::panicking() {
            tracing::error!("surface image was not presented")
        }
    }
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    format: Format,
    usage: ImageUsageFlags,
    mode: PresentMode,
    images: Vec<SwapchainImageState>,
    acquired_count: u32,
    optimal: bool,
}

struct SwapchainImageState {
    image: Image,
    acquire: Semaphore,
    release: Semaphore,
}

/// Swapchain properties.
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

    pub fn find_best_surface_format(&self) -> Option<Format> {
        const TARGET: Format = Format::BGRA8Srgb;
        const COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;

        let mut alternative_target = None;
        for &item in &self.surface_formats {
            let Some(format) = Format::from_vk(item.format) else {
                continue;
            };

            if format == TARGET && item.color_space == COLOR_SPACE {
                return Some(format);
            } else if alternative_target.is_none() && item.color_space == COLOR_SPACE {
                alternative_target = Some(format);
            }
        }

        alternative_target.or(self
            .surface_formats
            .iter()
            .find_map(|item| Format::from_vk(item.format)))
    }

    pub fn find_best_present_mode(&self) -> PresentMode {
        const TARGET: PresentMode = PresentMode::Mailbox;
        const FALLBACK: PresentMode = PresentMode::Fifo;

        self.present_modes
            .iter()
            .copied()
            .filter_map(PresentMode::try_from_vk)
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

static IMAGE_ID: AtomicU64 = AtomicU64::new(1);
