use std::collections::VecDeque;

use anyhow::{Context, Result};
use shared::util::WithDefer;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{KhrSurfaceExtension, KhrSwapchainExtension};
use winit::window::Window;

use crate::device::WeakDevice;

pub struct Surface {
    handle: vk::SurfaceKHR,
    owner: WeakDevice,
    swapchain: Option<Swapchain>,
    unused_swapchains: VecDeque<Swapchain>,
    swapchain_support: SwapchainSupport,
}

impl Surface {
    pub fn new(handle: vk::SurfaceKHR, device: &crate::device::Device) -> Result<Self> {
        let instance = device.graphics().instance();
        let swapchain_support = SwapchainSupport::new(instance, device.physical(), handle)?;

        anyhow::ensure!(
            !swapchain_support.supported_families.is_empty(),
            "no queues with present capability found"
        );

        Ok(Surface {
            handle,
            owner: device.downgrade(),
            swapchain: None,
            unused_swapchains: VecDeque::new(),
            swapchain_support,
        })
    }

    pub fn swapchain_support(&self) -> &SwapchainSupport {
        &self.swapchain_support
    }

    pub fn configure(
        &mut self,
        window: &Window,
        usage: vk::ImageUsageFlags,
        format: vk::Format,
        mode: vk::PresentModeKHR,
    ) -> Result<()> {
        let device = self.owner.upgrade().context("device was already dropped")?;
        let instance = device.graphics().instance();
        let logical = device.logical();

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

        let image_extent = self.swapchain_support.compute_swapchain_extent(window);

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

        // TODO: create images (move from base.rs)

        tracing::debug!(swapchain = ?*handle, "created swapchain");

        Ok(())
    }

    fn cleanup_unused_swapchains(&mut self, device: &crate::device::Device) {
        let logical = device.logical();

        while let Some(swapchain) = self.unused_swapchains.pop_front() {
            unsafe { logical.destroy_swapchain_khr(swapchain.handle, None) };
        }
    }
}

struct Swapchain {
    handle: vk::SwapchainKHR,
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
        for item in self.surface_formats {
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
