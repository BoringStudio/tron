use crate::types::VertexAttributeKind;
use crate::util::{RawResourceHandle, ResourceHandle};

pub type MaterialHandle = ResourceHandle<MaterialTag>;
pub(crate) type RawMaterialHandle = RawResourceHandle<MaterialTag>;

pub struct MaterialTag;

pub trait Material: Send + Sync + 'static {
    type ShaderDataType: gfx::Std430 + Send + Sync;
    type RequiredAttributes: MaterialArray<VertexAttributeKind>;
    type SupportedAttributes: MaterialArray<VertexAttributeKind>;

    fn required_attributes() -> Self::RequiredAttributes;
    fn supported_attributes() -> Self::SupportedAttributes;

    fn key(&self) -> u64;
    fn sorting(&self) -> Sorting;

    fn shader_data(&self) -> Self::ShaderDataType;
}

pub trait MaterialArray<T>: AsRef<[T]> + Clone {
    const LEN: usize;

    type U32Array: gfx::Std430 + std::fmt::Debug + Send + Sync;

    fn map_to_u32<F>(self, f: F) -> Self::U32Array
    where
        F: FnMut(T) -> u32;

    fn iter(self) -> impl Iterator<Item = T>;
}

impl<T, const N: usize> MaterialArray<T> for [T; N]
where
    T: Clone,
{
    const LEN: usize = N;

    type U32Array = [u32; N];

    #[inline]
    fn map_to_u32<F>(self, f: F) -> Self::U32Array
    where
        F: FnMut(T) -> u32,
    {
        self.map(f)
    }

    #[inline]
    fn iter(self) -> impl Iterator<Item = T> {
        self.into_iter()
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
