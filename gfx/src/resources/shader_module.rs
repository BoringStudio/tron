use std::borrow::Cow;
use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;
use crate::util::FromGfx;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct VertexShader {
    module: ShaderModule,
    entry: Cow<'static, str>,
}

impl VertexShader {
    pub fn module(&self) -> &ShaderModule {
        &self.module
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FragmentShader {
    module: ShaderModule,
    entry: Cow<'static, str>,
}

impl FragmentShader {
    pub fn module(&self) -> &ShaderModule {
        &self.module
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ComputeShader {
    module: ShaderModule,
    entry: Cow<'static, str>,
}

impl ComputeShader {
    pub fn module(&self) -> &ShaderModule {
        &self.module
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct ShaderStageFlags: u32 {
        const VERTEX = 1;
        const TESSELLATION_CONTROL = 1 << 1;
        const TESSELLATION_EVALUATION = 1 << 2;
        const GEOMETRY = 1 << 3;
        const FRAGMENT = 1 << 4;

        const COMPUTE = 1 << 5;

        const ALL_GRAPHICS = Self::VERTEX.bits()
            | Self::TESSELLATION_CONTROL.bits()
            | Self::TESSELLATION_EVALUATION.bits()
            | Self::GEOMETRY.bits()
            | Self::FRAGMENT.bits();

        const ALL = i32::MAX as u32;
    }
}

impl FromGfx<ShaderStageFlags> for vk::ShaderStageFlags {
    fn from_gfx(value: ShaderStageFlags) -> Self {
        let mut res = Self::empty();
        if value.contains(ShaderStageFlags::VERTEX) {
            res |= Self::VERTEX;
        }
        if value.contains(ShaderStageFlags::TESSELLATION_CONTROL) {
            res |= Self::TESSELLATION_CONTROL;
        }
        if value.contains(ShaderStageFlags::TESSELLATION_EVALUATION) {
            res |= Self::TESSELLATION_EVALUATION;
        }
        if value.contains(ShaderStageFlags::GEOMETRY) {
            res |= Self::GEOMETRY;
        }
        if value.contains(ShaderStageFlags::FRAGMENT) {
            res |= Self::FRAGMENT;
        }
        if value.contains(ShaderStageFlags::COMPUTE) {
            res |= Self::COMPUTE;
        }
        res
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl From<ShaderStage> for ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::Fragment => Self::FRAGMENT,
            ShaderStage::Compute => Self::COMPUTE,
        }
    }
}

impl FromGfx<ShaderStage> for vk::ShaderStageFlags {
    fn from_gfx(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::Fragment => Self::FRAGMENT,
            ShaderStage::Compute => Self::COMPUTE,
        }
    }
}

pub struct ShaderModuleInfo {
    pub data: Box<[u32]>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ShaderModule {
    inner: Arc<Inner>,
}

impl ShaderModule {
    pub(crate) fn new(handle: vk::ShaderModule, info: ShaderModuleInfo, owner: WeakDevice) -> Self {
        Self {
            inner: Arc::new(Inner {
                handle,
                info,
                owner,
            }),
        }
    }

    pub fn info(&self) -> &ShaderModuleInfo {
        &self.inner.info
    }

    pub fn handle(&self) -> vk::ShaderModule {
        self.inner.handle
    }
}

impl std::fmt::Debug for ShaderModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.debug_struct("ShaderModule")
                .field("handle", &self.inner.handle)
                .field("owner", &self.inner.owner)
                .finish()
        } else {
            std::fmt::Debug::fmt(&self.inner.handle, f)
        }
    }
}

impl Eq for ShaderModule {}
impl PartialEq for ShaderModule {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl std::hash::Hash for ShaderModule {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(&*self.inner, state)
    }
}

struct Inner {
    handle: vk::ShaderModule,
    info: ShaderModuleInfo,
    owner: WeakDevice,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(device) = self.owner.upgrade() {
            unsafe { device.destroy_shader_module(self.handle) }
        }
    }
}
