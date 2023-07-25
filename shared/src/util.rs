use std::mem::ManuallyDrop;

pub struct Defer<T, F: FnOnce(T)> {
    data: ManuallyDrop<T>,
    deleter: Option<F>,
}

impl<T, F: FnOnce(T)> Defer<T, F> {
    #[inline(always)]
    pub fn new(data: T, deleter: F) -> Self {
        Self {
            data: ManuallyDrop::new(data),
            deleter: Some(deleter),
        }
    }

    #[inline(always)]
    pub fn disarm(mut self) -> T {
        self.deleter = None;
        // SAFETY: this is the last place where deleter is called
        unsafe { ManuallyDrop::take(&mut self.data) }
    }
}

impl<T, F: FnOnce(T)> Drop for Defer<T, F> {
    fn drop(&mut self) {
        if let Some(f) = self.deleter.take() {
            // SAFETY: this is the last place where deleter is called
            let data = unsafe { ManuallyDrop::take(&mut self.data) };
            f(data);
        }
    }
}

impl<T, F: FnOnce(T)> std::ops::Deref for Defer<T, F> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.data
    }
}

impl<T, F: FnOnce(T)> std::ops::DerefMut for Defer<T, F> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

pub trait WithDefer<F: FnOnce(Self)>: Sized {
    fn with_defer(self, deleter: F) -> Defer<Self, F>;
}

impl<T, F: FnOnce(Self)> WithDefer<F> for T {
    #[inline(always)]
    fn with_defer(self, deleter: F) -> Defer<Self, F> {
        Defer::new(self, deleter)
    }
}
