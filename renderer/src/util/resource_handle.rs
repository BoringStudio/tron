use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub trait HandleAllocator<T: HandleData> {
    fn alloc(&self, deleter: Arc<T::Deleter>) -> ResourceHandle<T>;
    fn dealloc(&self, handle: RawResourceHandle<T>);
}

pub trait HandleData: Send + Sync + 'static {
    type Deleter: HandleDeleter<Self>;
}

pub trait HandleDeleter<T: ?Sized>: Send + Sync + 'static {
    fn delete(&self, handle: RawResourceHandle<T>);
}

pub struct SimpleHandleAllocator<T> {
    next: AtomicUsize,
    _phantom: PhantomData<T>,
}

impl<T> Default for SimpleHandleAllocator<T> {
    fn default() -> Self {
        Self {
            next: AtomicUsize::new(0),
            _phantom: PhantomData,
        }
    }
}

impl<T: HandleData> HandleAllocator<T> for SimpleHandleAllocator<T> {
    fn alloc(&self, deleter: Arc<T::Deleter>) -> ResourceHandle<T> {
        ResourceHandle {
            index: self.next.fetch_add(1, Ordering::Relaxed),
            refcount: deleter,
        }
    }

    fn dealloc(&self, _handle: RawResourceHandle<T>) {}
}

pub struct FreelistHandleAllocator<T> {
    next: AtomicUsize,
    free_list: Mutex<Vec<usize>>,
    _phantom: PhantomData<T>,
}

impl<T> Default for FreelistHandleAllocator<T> {
    fn default() -> Self {
        Self {
            next: AtomicUsize::new(0),
            free_list: Mutex::new(Vec::new()),
            _phantom: PhantomData,
        }
    }
}

impl<T: HandleData> HandleAllocator<T> for FreelistHandleAllocator<T> {
    fn alloc(&self, deleter: Arc<T::Deleter>) -> ResourceHandle<T> {
        let index = self
            .free_list
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| self.next.fetch_add(1, Ordering::Relaxed));

        ResourceHandle {
            index,
            refcount: deleter,
        }
    }

    fn dealloc(&self, handle: RawResourceHandle<T>) {
        self.free_list.lock().unwrap().push(handle.index);
    }
}

pub struct ResourceHandle<T: HandleData> {
    index: usize,
    refcount: Arc<T::Deleter>,
}

impl<T: HandleData> ResourceHandle<T> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn raw(&self) -> RawResourceHandle<T> {
        RawResourceHandle {
            index: self.index,
            _phantom: Default::default(),
        }
    }
}

impl<T: HandleData> Drop for ResourceHandle<T> {
    fn drop(&mut self) {
        if Arc::strong_count(&self.refcount) == 1 {
            self.refcount.delete(self.raw());
        }
    }
}

impl<T: HandleData> Clone for ResourceHandle<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            refcount: self.refcount.clone(),
        }
    }
}

impl<T: HandleData> Eq for ResourceHandle<T> {}
impl<T: HandleData> PartialEq for ResourceHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: HandleData> std::hash::Hash for ResourceHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state)
    }
}

impl<T: HandleData> std::fmt::Debug for ResourceHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceHandle")
            .field("id", &self.index)
            .field("refcount", &Arc::strong_count(&self.refcount))
            .finish()
    }
}

pub struct RawResourceHandle<T: ?Sized> {
    pub index: usize,
    _phantom: PhantomData<T>,
}

impl<T: ?Sized> Copy for RawResourceHandle<T> {}
impl<T: ?Sized> Clone for RawResourceHandle<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> std::fmt::Debug for RawResourceHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawResourceHandle")
            .field("id", &self.index)
            .finish()
    }
}

impl<T: ?Sized> Eq for RawResourceHandle<T> {}
impl<T: ?Sized> PartialEq for RawResourceHandle<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: ?Sized> std::hash::Hash for RawResourceHandle<T> {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.index, state)
    }
}
