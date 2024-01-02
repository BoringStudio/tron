use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use indexmap::IndexMap;

pub struct ResourceRegistry<T, H> {
    mapping: IndexMap<usize, ResourceStorage<T>, ahash::RandomState>,
    _handle_type: PhantomData<H>,
}

impl<T, H> ResourceRegistry<T, H> {
    pub fn new() -> Self {
        Self {
            mapping: Default::default(),
            _handle_type: Default::default(),
        }
    }

    pub fn insert(&mut self, handle: &ResourceHandle<H>, data: T) {
        self.mapping.insert(
            handle.id,
            ResourceStorage {
                weak_handle: handle.downgrade(),
                data,
            },
        );
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = &T> {
        self.mapping.values().map(|storage| &storage.data)
    }

    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut T> {
        self.mapping.values_mut().map(|storage| &mut storage.data)
    }

    pub fn get(&self, handle: RawResourceHandle<H>) -> &T {
        &self.mapping.get(&handle.id).unwrap().data
    }

    pub fn get_mut(&mut self, handle: &ResourceHandle<H>) -> &mut T {
        &mut self.mapping.get_mut(&handle.id).unwrap().data
    }
}

impl<T, H> Default for ResourceRegistry<T, H> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

struct ResourceStorage<T> {
    weak_handle: Weak<()>,
    data: T,
}

pub struct ResourceHandle<T> {
    id: usize,
    refcount: Arc<()>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ResourceHandle<T> {
    pub fn allocate(id: &AtomicUsize) -> Self {
        let id = id.fetch_add(1, Ordering::Relaxed);
        Self::new(id)
    }

    pub fn new(id: usize) -> Self {
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
        Self {
            id: self.id,
            _phantom: Default::default(),
        }
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

pub(crate) static RESOURCE_ID: AtomicUsize = AtomicUsize::new(0);
