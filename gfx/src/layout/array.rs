use super::{AsStd140, AsStd430, Padded, Std140, Std430};

// === std140 ===

unsafe impl<T: Std140, const N: usize> Std140 for [T; N] {
    // NOTE: `std140` requires arrays to be aligned to at least 16 bytes.
    const ALIGN_MASK: u64 = <T as Std140>::ALIGN_MASK | 0b1111;

    type ArrayPadding = [u8; 0];
}

impl<T, const N: usize> AsStd140 for [T; N]
where
    T: AsStd140,
{
    type Output = [Padded<T::Output, <T::Output as Std140>::ArrayPadding>; N];

    #[inline]
    fn write_as_std140(&self, dst: &mut Self::Output) {
        for (src, dst) in self.iter().zip(dst.iter_mut()) {
            src.write_as_std140(&mut dst.value);
        }
    }
}

// === std430 ===

unsafe impl<T: Std430, const N: usize> Std430 for [T; N] {
    // NOTE: `std430` arrays alignment is the same as its element alignment.
    const ALIGN_MASK: u64 = <T as Std430>::ALIGN_MASK;

    type ArrayPadding = [u8; 0];
}

impl<T, const N: usize> AsStd430 for [T; N]
where
    T: AsStd430,
{
    type Output = [Padded<T::Output, <T::Output as Std430>::ArrayPadding>; N];

    #[inline]
    fn write_as_std430(&self, dst: &mut Self::Output) {
        for (src, dst) in self.iter().zip(dst.iter_mut()) {
            src.write_as_std430(&mut dst.value);
        }
    }
}
