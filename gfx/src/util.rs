use vulkanalia::vk;

pub trait FromGfx<T>: Sized {
    fn from_gfx(value: T) -> Self;
}

pub trait ToVk<T>: Sized {
    fn to_vk(self) -> T;
}

pub trait TryFromVk<T>: Sized {
    fn try_from_vk(value: T) -> Option<Self>;
}

pub trait TryIntoGfx<T>: Sized {
    fn try_into_gfx(self) -> Option<T>;
}

impl<T, U> ToVk<U> for T
where
    U: FromGfx<T>,
{
    #[inline]
    fn to_vk(self) -> U {
        U::from_gfx(self)
    }
}

impl<T> FromGfx<T> for T {
    #[inline(always)]
    fn from_gfx(t: T) -> T {
        t
    }
}

impl<T, U> TryIntoGfx<U> for T
where
    U: TryFromVk<T>,
{
    #[inline]
    fn try_into_gfx(self) -> Option<U> {
        U::try_from_vk(self)
    }
}

impl<T> TryFromVk<T> for T {
    #[inline(always)]
    fn try_from_vk(t: T) -> Option<T> {
        Some(t)
    }
}

impl FromGfx<glam::IVec2> for vk::Offset2D {
    #[inline]
    fn from_gfx(value: glam::IVec2) -> Self {
        // SAFETY: both `glam::IVec2` and `vk::Offset2D` the
        // same layout guaranteed by `repr(C)`.
        unsafe { std::mem::transmute(value) }
    }
}

impl FromGfx<glam::IVec3> for vk::Offset3D {
    #[inline]
    fn from_gfx(value: glam::IVec3) -> Self {
        // SAFETY: both `glam::IVec3` and `vk::Offset3D` the
        // same layout guaranteed by `repr(C)`.
        unsafe { std::mem::transmute(value) }
    }
}

impl FromGfx<glam::UVec2> for vk::Extent2D {
    #[inline]
    fn from_gfx(value: glam::UVec2) -> Self {
        // SAFETY: both `glam::UVec2` and `vk::Extent2D` the
        // same layout guaranteed by `repr(C)`.
        unsafe { std::mem::transmute(value) }
    }
}

impl FromGfx<glam::UVec3> for vk::Extent3D {
    #[inline]
    fn from_gfx(value: glam::UVec3) -> Self {
        // SAFETY: both `glam::UVec3` and `vk::Extent3D` the
        // same layout guaranteed by `repr(C)`.
        unsafe { std::mem::transmute(value) }
    }
}
