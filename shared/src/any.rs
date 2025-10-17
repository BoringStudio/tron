pub struct AnyVec {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
    metadata: &'static VecMetadata,
}

impl AnyVec {
    pub fn new<T: Send + Sync>() -> Self {
        Self::from(Vec::<T>::new())
    }

    /// # Safety
    /// The following must be true:
    /// - `T` must be an original type of `Vec<T>`.
    pub unsafe fn typed_data<T>(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr.cast(), self.len)
    }

    /// # Safety
    /// The following must be true:
    /// - `T` must be an original type of `Vec<T>`.
    pub unsafe fn typed_data_mut<T>(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr.cast(), self.len)
    }

    /// Downcast to a mutable reference to a `Vec<T>`.
    ///
    /// # Safety
    /// The following must be true:
    /// - `T` must be an original type of `Vec<T>`.
    pub unsafe fn downcast_mut<T>(&mut self) -> AnyVecGuard<'_, T> {
        let vec = self.swap_vec(Vec::new());
        AnyVecGuard { vec, data: self }
    }

    /// # Safety
    /// The following must be true:
    /// - `T` must be an original type of `Vec<T>`.
    unsafe fn swap_vec<T>(&mut self, new: Vec<T>) -> Vec<T> {
        let mut new = std::mem::ManuallyDrop::new(new);
        let mut ptr = new.as_mut_ptr().cast();
        let mut length = new.len();
        let mut capacity = new.capacity();
        std::mem::swap(&mut self.ptr, &mut ptr);
        std::mem::swap(&mut self.len, &mut length);
        std::mem::swap(&mut self.capacity, &mut capacity);
        // SAFETY: these values came from us, and we always leave ourselves in
        // a valid state
        Vec::from_raw_parts(ptr.cast(), length, capacity)
    }
}

impl Drop for AnyVec {
    fn drop(&mut self) {
        // SAFETY:
        // - `self.ptr` was aquired from a `Vec<T>`.
        // - `self.len` is equal to `vec.len()`.
        // - `self.capacity` is equal to an original capacity of `Vec<T>`.
        unsafe { (self.metadata.drop_fn)(self.ptr, self.len, self.capacity) }
    }
}

impl<T: Send + Sync> From<Vec<T>> for AnyVec {
    fn from(vec: Vec<T>) -> Self {
        let mut vec = std::mem::ManuallyDrop::new(vec);
        let ptr = vec.as_mut_ptr().cast();
        let len = vec.len();
        let capacity = vec.capacity();
        let metadata = T::METADATA;

        Self {
            ptr,
            len,
            capacity,
            metadata,
        }
    }
}

// SAFETY: `AnyVec` can only be constructed from `Vec<T>` where `T: Send + Sync`.
unsafe impl Send for AnyVec {}
unsafe impl Sync for AnyVec {}

pub struct AnyVecGuard<'a, T> {
    vec: Vec<T>,
    data: &'a mut AnyVec,
}

impl<'a, T> Drop for AnyVecGuard<'a, T> {
    fn drop(&mut self) {
        // SAFETY: `T` is the same type as used to construct `self.data`.
        unsafe { self.data.swap_vec(std::mem::take(&mut self.vec)) };
    }
}

impl<T> std::ops::Deref for AnyVecGuard<'_, T> {
    type Target = Vec<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<T> std::ops::DerefMut for AnyVecGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

trait WithVecMetadata: Send + Sync {
    const METADATA: &'static VecMetadata;
}

impl<T: Send + Sync> WithVecMetadata for T {
    const METADATA: &'static VecMetadata = &VecMetadata {
        drop_fn: drop_vec::<T>,
    };
}

struct VecMetadata {
    drop_fn: unsafe fn(*mut u8, usize, usize),
}

/// # Safety
/// The following must be true:
/// - `ptr` must be aquired from a `Vec<T>`.
/// - `length` must be equal to `vec.len()`.
/// - `capacity` must be equal to an original capacity of `Vec<T>`.
unsafe fn drop_vec<T>(ptr: *mut u8, length: usize, capacity: usize) {
    Vec::<T>::from_raw_parts(ptr.cast(), length, capacity);
}
