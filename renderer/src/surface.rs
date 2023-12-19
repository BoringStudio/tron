use vulkanalia::prelude::v1_0::*;

pub struct Surface {
    handle: vk::SurfaceKHR,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
}
