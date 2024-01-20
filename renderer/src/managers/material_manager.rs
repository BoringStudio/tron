use std::any::TypeId;

use shared::FastHashMap;

use crate::managers::GpuMesh;
use crate::types::{Material, MaterialArray, RawMaterialHandle};

#[derive(Default)]
pub struct MaterialManager {
    handles: FastHashMap<RawMaterialHandle, HandleData>,
    archetypes: FastHashMap<TypeId, MaterialArchetype>,
}

impl MaterialManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<M: Material>(&mut self, handle: RawMaterialHandle, material: M) {
        let archetype = self.archetype_mut::<M>();

        let index = archetype.free_indices.pop().unwrap_or_else(|| {
            let index = archetype.next_index;
            archetype.next_index += 1;
            index
        });

        {
            // SAFETY: `downcast_mut` template parameter is the same as the one used to
            // construct `archetype`.
            let mut data = unsafe { archetype.data.downcast_mut::<Option<M>>() };
            if index >= data.len() {
                data.resize_with(index.next_power_of_two(), || None);
            }
            data[index] = Some(material);
        }

        self.handles.insert(
            handle,
            HandleData {
                archetype: TypeId::of::<M>(),
                index,
            },
        );
    }

    pub fn update<M: Material>(&mut self, handle: RawMaterialHandle, material: M) {
        let HandleData { archetype, index } = &self.handles[&handle];
        assert_eq!(*archetype, TypeId::of::<M>());

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        // SAFETY: `downcast_mut` template parameter is the same as the one used to
        // construct `archetype`.
        let mut data = unsafe { archetype.data.downcast_mut::<Option<M>>() };
        let item = data.get_mut(*index).expect("invalid handle index");
        *item.as_mut().expect("value was not initialized") = material;
    }

    pub fn remove(&mut self, handle: RawMaterialHandle) {
        let HandleData { archetype, index } = &self.handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.remove_data)(&mut archetype.data, handle, *index);
        archetype.free_indices.push(*index);
    }

    fn archetype_mut<M: Material>(&mut self) -> &mut MaterialArchetype {
        let id = TypeId::of::<M>();
        self.archetypes
            .entry(id)
            .or_insert_with(|| MaterialArchetype {
                data: MaterialData::new::<Option<M>>(),
                next_index: 0,
                free_indices: Vec::new(),
                remove_data: remove_data::<M>,
            })
    }
}

struct HandleData {
    archetype: TypeId,
    index: usize,
}

struct MaterialArchetype {
    data: MaterialData,
    next_index: usize,
    free_indices: Vec<usize>,
    remove_data: fn(&mut MaterialData, RawMaterialHandle, usize),
}

type FnAddObject = fn(&[u8], AddObjectArgs<'_>);

struct AddObjectArgs<'a> {
    mesh: &'a GpuMesh,
}

// TODO: add to archetype to abstract from mesh+data recording
#[allow(unused)]
fn record_object<M: Material>(_material: &M, args: AddObjectArgs<'_>) {
    let required_attributes_mask = M::required_attributes()
        .iter()
        .fold(0u8, |mask, attribute| mask | attribute as u8);
    let mesh_attributes_mask = args
        .mesh
        .attributes()
        .fold(0u8, |mask, attribute| mask | attribute as u8);

    assert_eq!(
        mesh_attributes_mask & required_attributes_mask,
        required_attributes_mask
    );

    let vertex_attribute_offsets = M::supported_attributes().map_to_u32(|attribute| {
        match args.mesh.get_attribute_range(attribute) {
            Some(range) => range.start as u32,
            None => u32::MAX,
        }
    });

    let indices = args.mesh.indices();
    let first_index = indices.start;
    let index_count = indices.end - indices.start;

    // TODO: add object
}

fn remove_data<M: Material>(data: &mut MaterialData, handle: RawMaterialHandle, index: usize) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to
    // construct `data`.
    let mut data = unsafe { data.downcast_mut::<Option<M>>() };
    let item = data.get_mut(index).expect("invalid handle index");
    std::mem::take(item).expect("value was not initialized");
}

struct MaterialData {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
    metadata: &'static VecMetadata,
}

impl MaterialData {
    fn new<T: Send + Sync>() -> Self {
        Self::from(Vec::<T>::new())
    }

    fn byte_len(&self) -> usize {
        self.len * self.metadata.item_size
    }

    fn untyped_data(&self) -> &[u8] {
        // SAFETY: `self.ptr` is a valid pointer to a slice of `self.byte_len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.byte_len()) }
    }

    /// # Safety
    /// The following must be true:
    /// - `T` must not be a ZST.
    /// - `T` must be an original type of `Vec<T>`.
    unsafe fn typed_data<T>(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr.cast(), self.len)
    }

    /// # Safety
    /// The following must be true:
    /// - `T` must not be a ZST.
    /// - `T` must be an original type of `Vec<T>`.
    unsafe fn downcast_mut<T>(&mut self) -> MaterialDataGuard<T> {
        let vec = self.swap_vec(Vec::new());
        MaterialDataGuard { vec, data: self }
    }

    /// # Safety
    /// The following must be true:
    /// - `T` must not be a ZST.
    /// - `T` must be an original type of `Vec<T>`.
    unsafe fn swap_vec<T>(&mut self, new: Vec<T>) -> Vec<T> {
        let mut new = std::mem::ManuallyDrop::new(new);
        let mut ptr = new.as_mut_ptr().cast();
        let mut length = new.len();
        let mut capacity = new.capacity();
        std::mem::swap(&mut self.ptr, &mut ptr);
        std::mem::swap(&mut self.len, &mut length);
        std::mem::swap(&mut self.capacity, &mut capacity);
        // SAFETY: these values came from us, and we always leave ourself in
        // a valid state
        Vec::from_raw_parts(ptr.cast(), length, capacity)
    }
}

impl Drop for MaterialData {
    fn drop(&mut self) {
        // SAFETY:
        // - `T` is not a ZST.
        // - `self.ptr` was aquired from a `Vec<T>`.
        // - `self.byte_len` is equal to `vec.len() * std::mem::size_of::<T>()`.
        // - `self.capacity` is equal to an original capacity of `Vec<T>`.
        unsafe { (self.metadata.drop_fn)(self.ptr, self.len, self.capacity) }
    }
}

impl<T: Send + Sync> From<Vec<T>> for MaterialData {
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

// SAFETY: `MaterialData` can only be constructed from `Vec<T>`
// where `T: Send + Sync`.
unsafe impl Send for MaterialData {}
unsafe impl Sync for MaterialData {}

struct MaterialDataGuard<'a, T> {
    vec: Vec<T>,
    data: &'a mut MaterialData,
}

impl<'a, T> Drop for MaterialDataGuard<'a, T> {
    fn drop(&mut self) {
        // SAFETY: `T` is not a ZST and is the same type as used to construct `self.data`.
        unsafe { self.data.swap_vec(std::mem::take(&mut self.vec)) };
    }
}

impl<T> std::ops::Deref for MaterialDataGuard<'_, T> {
    type Target = Vec<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<T> std::ops::DerefMut for MaterialDataGuard<'_, T> {
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
        item_size: std::mem::size_of::<T>(),
        drop_fn: drop_vec::<T>,
    };
}

struct VecMetadata {
    item_size: usize,
    drop_fn: unsafe fn(*mut u8, usize, usize),
}

/// # Safety
/// The following must be true:
/// - `T` must not be a ZST.
/// - `ptr` must be aquired from a `Vec<T>`.
/// - `bytes` must be equal to `vec.len() * std::mem::size_of::<T>()`.
/// - `capacity` must be equal to an original capacity of `Vec<T>`.
unsafe fn drop_vec<T>(ptr: *mut u8, length: usize, capacity: usize) {
    Vec::<T>::from_raw_parts(ptr.cast(), length, capacity);
}
