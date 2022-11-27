use std::hash::BuildHasherDefault;
use std::marker::PhantomData;
use std::sync::Weak;

use indexmap::map::IndexMap;
use rustc_hash::FxHasher;

use super::{RawResourceHandle, ResourceHandle};

pub struct ResourceRegistry<T, H> {
    mapping: IndexMap<usize, ResourceStorage<T>, BuildHasherDefault<FxHasher>>,
    _phantom: PhantomData<H>,
}

impl<T, H> ResourceRegistry<T, H> {
    pub fn new() -> Self {
        Self {
            mapping: IndexMap::with_hasher(Default::default()),
            _phantom: PhantomData,
        }
    }

    pub fn insert(&mut self, handle: &ResourceHandle<H>, data: T) {
        self.mapping.insert(
            handle.raw().id,
            ResourceStorage {
                refcount: handle.get_weak(),
                data,
            },
        )
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = &T> {
        self.mapping
            .values()
            .map(|ResourceStorage { data, .. }| data)
    }

    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut T> {
        self.mapping
            .values_mut()
            .map(|ResourceStorage { data, .. }| data)
    }

    pub fn get(&self, handle: RawResourceHandle<H>) -> &T {
        &self.mapping.get(&handle.id).unwrap().data
    }

    pub fn get_mut(&self, handle: RawResourceHandle<H>) -> &mut T {
        &mut self.mapping.get_mut(&handle.id).unwrap().data
    }
}

impl<T, H> Default for ResourceRegistry<T, H> {
    fn default() -> Self {
        Self::new()
    }
}

struct ResourceStorage<T> {
    refcount: Weak<()>,
    data: T,
}
