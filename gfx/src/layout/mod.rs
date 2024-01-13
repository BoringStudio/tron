use bytemuck::{Pod, Zeroable};

mod array;
mod matrix;
mod primitive;

/// A type that has a `std140` compatible layout.
///
/// # Safety
/// Must only be implemented for properly padded types.
pub unsafe trait Std140: Pod {
    const ALIGN_MASK: u64;

    type ArrayPadding: Padding;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// A type that can be converted to a [`Std140`] compatible type.
pub trait AsStd140 {
    type Output: Std140;

    fn as_std140(&self) -> Self::Output {
        let mut output = Zeroable::zeroed();
        self.write_as_std140(&mut output);
        output
    }

    fn write_as_std140(&self, dst: &mut Self::Output);
}

/// A type that has a `std430` compatible layout.
///
/// # Safety
/// Must only be implemented for properly padded types.
pub unsafe trait Std430: Pod {
    const ALIGN_MASK: u64;

    type ArrayPadding: Padding;

    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// A type that can be converted to a [`Std430`] compatible type.
pub trait AsStd430 {
    type Output: Std430;

    fn as_std430(&self) -> Self::Output {
        let mut output = Zeroable::zeroed();
        self.write_as_std430(&mut output);
        output
    }

    fn write_as_std430(&self, dst: &mut Self::Output);
}

/// A type that can be used as padding.
///
/// # Safety
/// Must have an alignment of 1.
pub unsafe trait Padding: Pod {}

unsafe impl<const N: usize> Padding for [u8; N] {}

/// A padded type.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Padded<T, P> {
    pub value: T,
    pub padding: P,
}

unsafe impl<T: Std140> Std140 for Padded<T, <T as Std140>::ArrayPadding> {
    const ALIGN_MASK: u64 = <T as Std140>::ALIGN_MASK;

    type ArrayPadding = [u8; 0];
}
unsafe impl<T: Std430> Std430 for Padded<T, <T as Std430>::ArrayPadding> {
    const ALIGN_MASK: u64 = <T as Std430>::ALIGN_MASK;

    type ArrayPadding = [u8; 0];
}

unsafe impl<T: Zeroable, P: Zeroable> Zeroable for Padded<T, P> {}
unsafe impl<T: Pod, P: Pod> Pod for Padded<T, P> {}

impl<T: std::fmt::Debug, P> std::fmt::Debug for Padded<T, P> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Padded")
            .field("value", &self.value)
            .field("padding", &std::mem::size_of::<P>())
            .finish()
    }
}

impl<T: PartialEq, P> Eq for Padded<T, P> {}
impl<T: PartialEq, P> PartialEq for Padded<T, P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Ord, P> Ord for Padded<T, P> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T: PartialOrd, P> PartialOrd for Padded<T, P> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: std::hash::Hash, P> std::hash::Hash for Padded<T, P> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_std140_repr() {
        type Repr<T> = <T as AsStd140>::Output;

        // bool stuff
        assert_eq!(<Repr<bool> as Std140>::ALIGN_MASK, 0b11);
        assert_eq!(std::mem::size_of::<Repr<bool>>(), 4);

        assert_eq!(<Repr<glam::BVec2> as Std140>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<glam::BVec2>>(), 8);

        assert_eq!(<Repr<glam::BVec3> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::BVec3>>(), 12);

        assert_eq!(<Repr<glam::BVec4> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::BVec4>>(), 16);

        // bool array stuff
        assert_eq!(Repr::<[bool; 4]>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[bool; 4]>>(), 64); // bool -> pad to 16 bytes

        assert_eq!(Repr::<[glam::BVec2; 4]>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::BVec2; 4]>>(), 64); // BVec2 -> pad to 16 bytes

        assert_eq!(<Repr::<[glam::BVec3; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::BVec3; 4]>>(), 64); // BVec3 -> pad to 16 bytes

        assert_eq!(<Repr::<[glam::BVec4; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::BVec4; 4]>>(), 64); // BVec4 -> pad to 16 bytes

        // float stuff
        assert_eq!(<Repr<f32> as Std140>::ALIGN_MASK, 0b11);
        assert_eq!(std::mem::size_of::<Repr<f32>>(), 4);

        assert_eq!(<Repr<glam::Vec2> as Std140>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<glam::Vec2>>(), 8);

        assert_eq!(<Repr<glam::Vec3> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Vec3>>(), 12);

        assert_eq!(<Repr<glam::Vec4> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Vec4>>(), 16);

        // float array stuff
        assert_eq!(Repr::<[f32; 4]>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[f32; 4]>>(), 64); // f32 -> pad to 16 bytes

        assert_eq!(Repr::<[glam::Vec2; 4]>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::Vec2; 4]>>(), 64); // Vec2 -> pad to 16 bytes

        assert_eq!(<Repr::<[glam::Vec3; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::Vec3; 4]>>(), 64); // Vec3 -> pad to 16 bytes

        assert_eq!(<Repr::<[glam::Vec4; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::Vec4; 4]>>(), 64);

        assert_eq!(<Repr::<[glam::DVec2; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::DVec2; 4]>>(), 64);

        assert_eq!(<Repr::<[glam::DVec4; 4]> as Std140>::ALIGN_MASK, 0b11111);
        assert_eq!(std::mem::size_of::<Repr<[glam::DVec4; 4]>>(), 128);

        // matrix stuff
        assert_eq!(<Repr<glam::Mat2> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Mat2>>(), 32); // Vec2 -> pad to 16 bytes

        assert_eq!(<Repr<glam::Mat3> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Mat3>>(), 48); // Vec3 -> pad to 16 bytes

        assert_eq!(<Repr<glam::Mat4> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Mat4>>(), 64);

        assert_eq!(<Repr<glam::Affine2> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Affine2>>(), 48); // Vec2 -> pad to 16 bytes

        assert_eq!(<Repr<glam::Affine3A> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Affine3A>>(), 64); // Vec3 -> pad to 16 bytes
    }

    #[test]
    fn correct_std430_repr() {
        type Repr<T> = <T as AsStd430>::Output;

        // bool stuff
        assert_eq!(<Repr<bool> as Std430>::ALIGN_MASK, 0b11);
        assert_eq!(std::mem::size_of::<Repr<bool>>(), 4);

        assert_eq!(<Repr<glam::BVec2> as Std430>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<glam::BVec2>>(), 8);

        assert_eq!(<Repr<glam::BVec3> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::BVec3>>(), 12);

        assert_eq!(<Repr<glam::BVec4> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::BVec4>>(), 16);

        // bool array stuff
        assert_eq!(Repr::<[bool; 4]>::ALIGN_MASK, 0b11);
        assert_eq!(std::mem::size_of::<Repr<[bool; 4]>>(), 16);

        assert_eq!(Repr::<[glam::BVec2; 4]>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<[glam::BVec2; 4]>>(), 32);

        assert_eq!(<Repr::<[glam::BVec3; 4]> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::BVec3; 4]>>(), 64); // BVec3 -> pad to 4 bytes

        assert_eq!(<Repr::<[glam::BVec4; 4]> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::BVec4; 4]>>(), 64);

        // float stuff
        assert_eq!(<Repr<f32> as Std430>::ALIGN_MASK, 0b11);
        assert_eq!(std::mem::size_of::<Repr<f32>>(), 4);

        assert_eq!(<Repr<glam::Vec2> as Std430>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<glam::Vec2>>(), 8);

        assert_eq!(<Repr<glam::Vec3> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Vec3>>(), 12);

        assert_eq!(<Repr<glam::Vec4> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Vec4>>(), 16);

        // float array stuff
        assert_eq!(Repr::<[f32; 4]>::ALIGN_MASK, 0b11);
        assert_eq!(std::mem::size_of::<Repr<[f32; 4]>>(), 16);

        assert_eq!(Repr::<[glam::Vec2; 4]>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<[glam::Vec2; 4]>>(), 32);

        assert_eq!(<Repr::<[glam::Vec3; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::Vec3; 4]>>(), 64); // Vec3 -> pad to 16 bytes

        assert_eq!(<Repr::<[glam::Vec4; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::Vec4; 4]>>(), 64);

        assert_eq!(<Repr::<[glam::DVec2; 4]> as Std140>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<[glam::DVec2; 4]>>(), 64);

        assert_eq!(<Repr::<[glam::DVec4; 4]> as Std140>::ALIGN_MASK, 0b11111);
        assert_eq!(std::mem::size_of::<Repr<[glam::DVec4; 4]>>(), 128);

        // matrix stuff
        assert_eq!(<Repr<glam::Mat2> as Std430>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<glam::Mat2>>(), 16);

        assert_eq!(<Repr<glam::Mat3> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Mat3>>(), 48); // Vec3 -> pad to 16 bytes

        assert_eq!(<Repr<glam::Mat4> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Mat4>>(), 64);

        assert_eq!(<Repr<glam::Affine2> as Std430>::ALIGN_MASK, 0b111);
        assert_eq!(std::mem::size_of::<Repr<glam::Affine2>>(), 24);

        assert_eq!(<Repr<glam::Affine3A> as Std430>::ALIGN_MASK, 0b1111);
        assert_eq!(std::mem::size_of::<Repr<glam::Affine3A>>(), 64); // Vec3 -> pad to 16 bytes
    }
}
