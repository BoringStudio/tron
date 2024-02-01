use crate::types::VertexAttributeKind;
use crate::util::{RawResourceHandle, ResourceHandle};

pub type MaterialInstanceHandle = ResourceHandle<MaterialInstanceTag>;
pub(crate) type RawMaterialInstanceHandle = RawResourceHandle<MaterialInstanceTag>;

pub struct MaterialInstanceTag;

pub trait MaterialInstance: Send + Sync + 'static {
    type ShaderDataType: gfx::Std430 + Send + Sync;
    type RequiredAttributes: VertexAttributeArray;
    type SupportedAttributes: VertexAttributeArray;

    fn required_attributes() -> Self::RequiredAttributes;
    fn supported_attributes() -> Self::SupportedAttributes;

    fn key(&self) -> u64;
    fn sorting(&self) -> Sorting;

    fn shader_data(&self) -> Self::ShaderDataType;
}

pub trait VertexAttributeArray: AsRef<[VertexAttributeKind]> + Clone {
    const LEN: usize;

    type U32Array: gfx::Std430 + std::fmt::Debug + Send + Sync;

    fn map_to_u32<F>(self, f: F) -> Self::U32Array
    where
        F: FnMut(VertexAttributeKind) -> u32;
}

impl<const N: usize> VertexAttributeArray for [VertexAttributeKind; N] {
    const LEN: usize = N;

    type U32Array = [u32; N];

    #[inline]
    fn map_to_u32<F>(self, f: F) -> Self::U32Array
    where
        F: FnMut(VertexAttributeKind) -> u32,
    {
        self.map(f)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Sorting {
    pub reason: SortingReason,
    pub order: SortingOrder,
}

impl Sorting {
    pub const OPAQUE: Self = Self {
        reason: SortingReason::Optimization,
        order: SortingOrder::FrontToBack,
    };

    pub const BLENDING: Self = Self {
        reason: SortingReason::Requirement,
        order: SortingOrder::BackToFront,
    };
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SortingOrder {
    FrontToBack,
    BackToFront,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SortingReason {
    Optimization,
    Requirement,
}
