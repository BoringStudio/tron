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
        self.free_list.lock().unwrap().push(handle.id);
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
    id: usize,
    refcount: Arc<()>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ResourceHandle<T> {
    fn new(id: usize) -> Self {
        Self {
            id,
            refcount: Default::default(),
            _phantom: Default::default(),
        }
    }

    pub fn raw(&self) -> RawResourceHandle<T> {
        RawResourceHandle {
            id: self.id,
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
            id: self.id,
            refcount: self.refcount.clone(),
            _phantom: self._phantom,
        }
    }
}

impl<T> Eq for ResourceHandle<T> {}
impl<T> PartialEq for ResourceHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> std::hash::Hash for ResourceHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T> std::fmt::Debug for ResourceHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceHandle")
            .field("id", &self.id)
            .field("refcount", &Arc::strong_count(&self.refcount))
            .finish()
    }
}

pub struct RawResourceHandle<T> {
    pub id: usize,
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
            .field("id", &self.id)
            .finish()
    }
}

impl<T> Eq for RawResourceHandle<T> {}
impl<T> PartialEq for RawResourceHandle<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
