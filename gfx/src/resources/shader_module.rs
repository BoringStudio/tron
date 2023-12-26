use std::borrow::Cow;
use std::sync::Arc;

use vulkanalia::prelude::v1_0::*;

use crate::device::WeakDevice;

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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
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
pub struct ShaderModule {
    inner: Arc<Inner>,
}

impl ShaderModule {
    pub fn new(handle: vk::ShaderModule, info: ShaderModuleInfo, owner: WeakDevice) -> Self {
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
