use bytemuck::Pod;
use glam::{
    DVec2, DVec3, DVec4, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec3A, Vec4,
};

use super::{AsStd140, AsStd430, Padding, Std140, Std430};

/// A simple native shader type.
pub trait PrimitiveShaderType: Pod {
    const ALIGN_MASK: u64;

    type ArrayPaddingStd140Type: Padding;
    type ArrayPaddingStd430Type: Padding;
}

// === std140 ===

unsafe impl<T: PrimitiveShaderType> Std140 for T {
    const ALIGN_MASK: u64 = <T as PrimitiveShaderType>::ALIGN_MASK;
    type ArrayPadding = <T as PrimitiveShaderType>::ArrayPaddingStd140Type;
}

impl<T: PrimitiveShaderType> AsStd140 for T {
    type Output = Self;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        *self
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        *dst = *self;
    }
}

// === std430 ===

unsafe impl<T: PrimitiveShaderType> Std430 for T {
    const ALIGN_MASK: u64 = <T as PrimitiveShaderType>::ALIGN_MASK;
    type ArrayPadding = <T as PrimitiveShaderType>::ArrayPaddingStd430Type;
}

impl<T: PrimitiveShaderType> AsStd430 for T {
    type Output = Self;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        *self
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        *dst = *self;
    }
}

// === bool and bveci ===

impl AsStd140 for bool {
    type Output = u32;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        *self as u32
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        *dst = *self as u32;
    }
}

impl AsStd430 for bool {
    type Output = u32;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        *self as u32
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        *dst = *self as u32;
    }
}

impl AsStd140 for glam::BVec2 {
    type Output = UVec2;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        UVec2::new(self.x as u32, self.y as u32)
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst.x = self.x as u32;
        dst.y = self.y as u32;
    }
}

impl AsStd430 for glam::BVec2 {
    type Output = UVec2;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        UVec2::new(self.x as u32, self.y as u32)
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst.x = self.x as u32;
        dst.y = self.y as u32;
    }
}

impl AsStd140 for glam::BVec3 {
    type Output = UVec3;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        UVec3::new(self.x as u32, self.y as u32, self.z as u32)
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst.x = self.x as u32;
        dst.y = self.y as u32;
        dst.z = self.z as u32;
    }
}

impl AsStd430 for glam::BVec3 {
    type Output = UVec3;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        UVec3::new(self.x as u32, self.y as u32, self.z as u32)
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst.x = self.x as u32;
        dst.y = self.y as u32;
        dst.z = self.z as u32;
    }
}

impl AsStd140 for glam::BVec4 {
    type Output = UVec4;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        UVec4::new(self.x as u32, self.y as u32, self.z as u32, self.w as u32)
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst.x = self.x as u32;
        dst.y = self.y as u32;
        dst.z = self.z as u32;
        dst.w = self.w as u32;
    }
}

impl AsStd430 for glam::BVec4 {
    type Output = UVec4;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        UVec4::new(self.x as u32, self.y as u32, self.z as u32, self.w as u32)
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst.x = self.x as u32;
        dst.y = self.y as u32;
        dst.z = self.z as u32;
        dst.w = self.w as u32;
    }
}

// === vec3a ===

impl AsStd140 for Vec3A {
    type Output = Vec3;

    #[inline]
    fn as_std140(&self) -> Self::Output {
        Vec3::from(*self)
    }

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        *dst = Vec3::from(*self);
    }
}

impl AsStd430 for Vec3A {
    type Output = Vec3;

    #[inline]
    fn as_std430(&self) -> Self::Output {
        Vec3::from(*self)
    }

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        *dst = Vec3::from(*self);
    }
}

// === other primitive types ===

macro_rules! impl_shader_type {
    ($(
        $ty:ty {
            align_mask: $align_mas:literal,
            array_padding_std140: $array_padding_std140:literal,
            array_padding_std430: $array_padding_std430:literal,
        }
    ),*$(,)?) => {$(
        impl PrimitiveShaderType for $ty {
            const ALIGN_MASK: u64 = $align_mas;

            type ArrayPaddingStd140Type = [u8; $array_padding_std140];
            type ArrayPaddingStd430Type = [u8; $array_padding_std430];
        }
    )*};
}

impl_shader_type!(
    // scalar
    u32 {
        align_mask: 0b11,
        array_padding_std140: 12,
        array_padding_std430: 0,
    },
    i32 {
        align_mask: 0b11,
        array_padding_std140: 12,
        array_padding_std430: 0,
    },
    f32 {
        align_mask: 0b11,
        array_padding_std140: 12,
        array_padding_std430: 0,
    },
    f64 {
        align_mask: 0b111,
        array_padding_std140: 8,
        array_padding_std430: 0,
    },
    // vec2n
    Vec2 {
        align_mask: 0b111, // align to 8 bytes
        array_padding_std140: 8,
        array_padding_std430: 0,
    },
    UVec2 {
        align_mask: 0b111, // align to  8 bytes
        array_padding_std140: 8,
        array_padding_std430: 0,
    },
    IVec2 {
        align_mask: 0b111, // align to  8 bytes
        array_padding_std140: 8,
        array_padding_std430: 0,
    },
    DVec2 {
        align_mask: 0b1111, // align to  16 bytes
        array_padding_std140: 0,
        array_padding_std430: 0,
    },
    // vec3n
    Vec3 {
        align_mask: 0b1111, // align to  16 bytes
        array_padding_std140: 4,
        array_padding_std430: 4,
    },
    UVec3 {
        align_mask: 0b1111, // align to  16 bytes
        array_padding_std140: 4,
        array_padding_std430: 4,
    },
    IVec3 {
        align_mask: 0b1111, // align to  16 bytes
        array_padding_std140: 4,
        array_padding_std430: 4,
    },
    DVec3 {
        align_mask: 0b11111, // align to 32 bytes
        array_padding_std140: 8,
        array_padding_std430: 8,
    },
    // vec4n
    Vec4 {
        align_mask: 0b1111, // align to 16 bytes
        array_padding_std140: 0,
        array_padding_std430: 0,
    },
    UVec4 {
        align_mask: 0b1111, // align to 16 bytes
        array_padding_std140: 0,
        array_padding_std430: 0,
    },
    IVec4 {
        align_mask: 0b1111, // align to 16 bytes
        array_padding_std140: 0,
        array_padding_std430: 0,
    },
    DVec4 {
        align_mask: 0b11111, // align to 32 bytes
        array_padding_std140: 0,
        array_padding_std430: 0,
    },
);
