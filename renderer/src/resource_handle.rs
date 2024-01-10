use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

pub struct ResourceHandleAllocator<T> {
    next: AtomicUsize,
    free_list: Mutex<Vec<usize>>,
    _phantom: PhantomData<T>,
}

impl<T> ResourceHandleAllocator<T> {
    pub fn alloc(&self) -> ResourceHandle<T> {
        let id = self
            .free_list
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| self.next.fetch_add(1, Ordering::Relaxed));

        ResourceHandle::new(id)
    }

    pub fn dealloc(&self, handle: &ResourceHandle<T>) {
        self.free_list.lock().unwrap().push(handle.index);
    }
}

impl<T> Default for ResourceHandleAllocator<T> {
    fn default() -> Self {
        Self {
            next: AtomicUsize::new(0),
            free_list: Mutex::new(Vec::new()),
            _phantom: PhantomData,
        }
    }
}

pub struct ResourceHandle<T> {
    index: usize,
    refcount: Arc<()>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ResourceHandle<T> {
    fn new(id: usize) -> Self {
        Self {
            index: id,
            refcount: Default::default(),
            _phantom: Default::default(),
        }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn raw(&self) -> RawResourceHandle<T> {
        RawResourceHandle {
            index: self.index,
            _phantom: Default::default(),
        }
    }

    fn downgrade(&self) -> Weak<()> {
        Arc::downgrade(&self.refcount)
    }
}

impl<T> Clone for ResourceHandle<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            refcount: self.refcount.clone(),
            _phantom: self._phantom,
        }
    }
}

impl<T> Eq for ResourceHandle<T> {}
impl<T> PartialEq for ResourceHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> std::hash::Hash for ResourceHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state)
    }
}

impl<T> std::fmt::Debug for ResourceHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceHandle")
            .field("id", &self.index)
            .field("refcount", &Arc::strong_count(&self.refcount))
            .finish()
    }
}

pub struct RawResourceHandle<T> {
    pub index: usize,
    _phantom: PhantomData<T>,
}

impl<T> Copy for RawResourceHandle<T> {}
impl<T> Clone for RawResourceHandle<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> std::fmt::Debug for RawResourceHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawResourceHandle")
            .field("id", &self.index)
            .finish()
    }
}

impl<T> Eq for RawResourceHandle<T> {}
impl<T> PartialEq for RawResourceHandle<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
