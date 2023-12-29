use bumpalo::Bump;

pub(crate) use self::access::*;
pub(crate) use self::traits::*;

mod access;
mod traits;

pub(crate) struct DeallocOnDrop<'a>(pub &'a mut Bump);

impl Drop for DeallocOnDrop<'_> {
    fn drop(&mut self) {
        self.0.reset();
    }
}

impl std::ops::Deref for DeallocOnDrop<'_> {
    type Target = Bump;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl std::ops::DerefMut for DeallocOnDrop<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}
