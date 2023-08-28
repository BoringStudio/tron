use std::rc::Rc;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::window::Window;

use super::base::{RendererBase, SwapchainSupport};

pub struct Swapchain {
    base: Rc<RendererBase>,
    swapchain: vk::SwapchainKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    image_fence_handles: Vec<vk::Fence>,
}

impl Swapchain {
    pub unsafe fn uninit(base: Rc<RendererBase>) -> Self {
        Self {
            base,
            swapchain: vk::SwapchainKHR::null(),
            swapchain_extent: Default::default(),
            swapchain_images: Default::default(),
            swapchain_image_views: Default::default(),
            image_fence_handles: Default::default(),
        }
    }

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn len(&self) -> usize {
        self.swapchain_image_views.len()
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.swapchain_image_views
    }

    pub unsafe fn make_framebuffers(
        &self,
        render_pass_handle: vk::RenderPass,
    ) -> Result<Vec<SwapchainFramebuffer>> {
        self.swapchain_image_views
            .iter()
            .map(|image_view| {
                SwapchainFramebuffer::new(
                    self.base.clone(),
                    render_pass_handle,
                    *image_view,
                    self.swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()
    }

    pub unsafe fn wait_for_image_fence(
        &mut self,
        index: usize,
        fence_handle: vk::Fence,
    ) -> Result<()> {
        let image_in_flight = self.image_fence_handles[index];
        if !image_in_flight.is_null() {
            self.base
                .device()
                .wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.image_fence_handles[index] = fence_handle;
        Ok(())
    }

    pub unsafe fn recreate(&mut self, window: &Window) -> Result<()> {
        if !self.swapchain.is_null() {
            self.base
                .device()
                .destroy_swapchain_khr(self.swapchain, None);
        }

        let device = self.base.device();
        let physical_device = self.base.physical_device();

        let swapchain_support = &physical_device.swapchain_support;
        let extent = compute_swapchain_extent(&swapchain_support, window);

        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count != 0
            && image_count > swapchain_support.capabilities.max_image_count
        {
            image_count = swapchain_support.capabilities.max_image_count;
        }

        let queue_indices = [
            physical_device.graphics_queue_family_idx,
            physical_device.present_queue_family_idx,
        ];

        let (image_sharing_mode, queue_indices) = if queue_indices[0] != queue_indices[1] {
            (vk::SharingMode::CONCURRENT, queue_indices.as_slice())
        } else {
            (vk::SharingMode::EXCLUSIVE, [].as_slice())
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.base.surface())
            .min_image_count(image_count)
            .image_format(swapchain_support.surface_format.format)
            .image_color_space(swapchain_support.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_indices)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(swapchain_support.present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        self.swapchain = device.create_swapchain_khr(&info, None)?;
        self.swapchain_images = device.get_swapchain_images_khr(self.swapchain)?;
        self.swapchain_extent = extent;

        let components = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        for image_view in self.swapchain_image_views.drain(..) {
            self.base.device().destroy_image_view(image_view, None);
        }
        self.swapchain_image_views.clear();
        for image in &self.swapchain_images {
            let info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::_2D)
                .format(swapchain_support.surface_format.format)
                .components(components)
                .subresource_range(subresource_range);

            self.swapchain_image_views
                .push(device.create_image_view(&info, None)?);
        }

        self.image_fence_handles.clear();
        self.image_fence_handles
            .resize_with(self.swapchain_image_views.len(), vk::Fence::null);

        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            let device = self.base.device();

            for image_view in self.swapchain_image_views.drain(..) {
                device.destroy_image_view(image_view, None);
            }

            if !self.swapchain.is_null() {
                device.destroy_swapchain_khr(self.swapchain, None);
            }
        }
    }
}

pub struct SwapchainFramebuffer {
    base: Rc<RendererBase>,
    handle: vk::Framebuffer,
}

impl SwapchainFramebuffer {
    pub unsafe fn new(
        base: Rc<RendererBase>,
        render_pass: vk::RenderPass,
        image_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&image_view))
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let handle = base.device().create_framebuffer(&info, None)?;

        Ok(Self { base, handle })
    }

    pub fn handle(&self) -> vk::Framebuffer {
        self.handle
    }
}

impl Drop for SwapchainFramebuffer {
    fn drop(&mut self) {
        unsafe {
            self.base.device().destroy_framebuffer(self.handle, None);
        }
    }
}

fn compute_swapchain_extent(swapchain_support: &SwapchainSupport, window: &Window) -> vk::Extent2D {
    let capabilities = &swapchain_support.capabilities;

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
