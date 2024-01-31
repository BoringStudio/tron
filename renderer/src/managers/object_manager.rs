use std::any::TypeId;
use std::collections::hash_map;

use anyhow::Result;
use gfx::AsStd430;
use glam::{Mat4, UVec4, Vec4};
use shared::{AnyVec, FastHashMap};

use crate::managers::{GpuMesh, MaterialManager, MeshManagerDataGuard};
use crate::types::{
    Material, MaterialArray, MaterialHandle, MeshHandle, RawStaticObjectHandle, StaticObject,
    VertexAttributeKind,
};
use crate::util::{
    BindlessResources, BoundingSphere, FreelistDoubleBuffer, ScatterCopy, StorageBufferHandle,
};

#[derive(Default)]
pub struct ObjectManager {
    static_handles: FastHashMap<RawStaticObjectHandle, HandleData>,
    archetypes: FastHashMap<TypeId, ObjectArchetype>,
}

impl ObjectManager {
    pub fn iter_static<M: Material>(
        &self,
    ) -> Option<StaticObjectsIter<'_, M::SupportedAttributes>> {
        let archetype = self.archetypes.get(&TypeId::of::<M>())?;

        // SAFETY: `typed_data` template parameter is the same as the one used to
        // construct `archetype`.
        let data = unsafe {
            archetype
                .data
                .typed_data::<SlotData<M::SupportedAttributes>>()
        };

        Some(StaticObjectsIter {
            inner: data.iter(),
            buffer_handle: archetype.buffer.handle(),
            slot: 0,
            len: archetype.active_object_count,
        })
    }

    #[tracing::instrument(level = "debug", name = "insert_object", skip_all)]
    pub fn insert_static(
        &mut self,
        handle: RawStaticObjectHandle,
        object: Box<StaticObject>,
        mesh_manager_data: &MeshManagerDataGuard,
        material_manager: &mut MaterialManager,
    ) {
        let mesh = mesh_manager_data[object.mesh.index()]
            .as_ref()
            .expect("invalid mesh handle");

        material_manager.write_static_object(
            object.material.raw(),
            WriteStaticObject {
                mesh,
                handle,
                object,
                object_manager: Some(self),
            },
        );
    }

    #[tracing::instrument(level = "debug", name = "update_object", skip_all)]
    pub fn update_static(&mut self, handle: RawStaticObjectHandle, transform: &Mat4) {
        let HandleData { archetype, slot } = &self.static_handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.update_transform)(archetype, *slot, transform);
    }

    #[tracing::instrument(level = "debug", name = "remove_object", skip_all)]
    pub fn remove_static(&mut self, handle: RawStaticObjectHandle) {
        let HandleData { archetype, slot } = &self.static_handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.remove)(archetype, *slot);
    }

    #[tracing::instrument(level = "debug", name = "flush_objects", skip_all)]
    pub fn flush(
        &mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        scatter_copy: &ScatterCopy,
        bindless_resources: &BindlessResources,
    ) -> Result<()> {
        for archetype in self.archetypes.values_mut() {
            (archetype.flush)(
                archetype,
                FlushObject {
                    device,
                    encoder,
                    scatter_copy,
                    bindless_resources,
                },
            )?;
        }
        Ok(())
    }

    fn get_or_create_archetype<M: Material>(&mut self) -> &mut ObjectArchetype {
        let id = TypeId::of::<M>();
        match self.archetypes.entry(id) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => entry.insert(ObjectArchetype {
                data: AnyVec::new::<SlotData<M::SupportedAttributes>>(),
                buffer: FreelistDoubleBuffer::with_capacity(INITIAL_BUFFER_CAPACITY),
                active_object_count: 0,
                next_slot: 0,
                free_slots: Vec::new(),
                flush: flush::<M::SupportedAttributes>,
                update_transform: update_transform::<M::SupportedAttributes>,
                remove: remove::<M::SupportedAttributes>,
            }),
        }
    }
}

const INITIAL_BUFFER_CAPACITY: u32 = 16;

struct HandleData {
    archetype: TypeId,
    slot: u32,
}

struct ObjectArchetype {
    data: AnyVec,
    buffer: FreelistDoubleBuffer,
    active_object_count: u32,
    next_slot: u32,
    free_slots: Vec<u32>,
    flush: fn(&mut ObjectArchetype, FlushObject) -> Result<()>,
    update_transform: fn(&mut ObjectArchetype, u32, &Mat4),
    remove: fn(&mut ObjectArchetype, u32),
}

type SlotData<A> = Option<GpuObject<<A as MaterialArray<VertexAttributeKind>>::U32Array>>;

pub struct GpuObject<A> {
    // NOTE: having `Some` here means that the object is enabled.
    // This is used to drop handles when the object is removed.
    pub enabled_object_data: Option<EnabledObjectData>,
    pub mesh_bounding_sphere: BoundingSphere,

    pub transform: Mat4,
    pub global_bounding_sphere: BoundingSphere,
    pub vertex_attribute_offsets: A,
    pub first_index: u32,
    pub index_count: u32,
    pub material_slot: u32,
}

struct EnabledObjectData {
    pub mesh_handle: MeshHandle,
    pub material_handle: MaterialHandle,
}

impl<A> GpuObject<A> {
    fn make_data(&self) -> UVec4 {
        glam::uvec4(
            self.first_index,
            self.index_count,
            self.material_slot,
            self.enabled_object_data.is_some() as _,
        )
    }
}

impl<A> gfx::AsStd430 for GpuObject<A>
where
    A: gfx::Std430,
{
    type Output = Std430GpuObject<A>;

    fn as_std430(&self) -> Self::Output {
        Std430GpuObject {
            transform: self.transform,
            transform_inverse_transpose: self.transform.inverse().transpose(),
            global_bounding_sphere: self.global_bounding_sphere.into(),
            data: self.make_data(),
            vertex_attribute_offsets: self.vertex_attribute_offsets,
        }
    }

    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst.transform = self.transform;
        dst.transform_inverse_transpose = self.transform.inverse().transpose();
        dst.global_bounding_sphere = self.global_bounding_sphere.into();
        dst.data = self.make_data();
        dst.vertex_attribute_offsets = self.vertex_attribute_offsets;
    }
}

#[derive(Clone, Copy)]
pub struct Std430GpuObject<A> {
    transform: Mat4,
    transform_inverse_transpose: Mat4,
    global_bounding_sphere: Vec4,
    data: UVec4,
    vertex_attribute_offsets: A,
}

unsafe impl<A: bytemuck::Pod> bytemuck::Pod for Std430GpuObject<A> {}
unsafe impl<A: bytemuck::Zeroable> bytemuck::Zeroable for Std430GpuObject<A> {}

unsafe impl<A: gfx::Std430> gfx::Std430 for Std430GpuObject<A> {
    const ALIGN_MASK: u64 = 0b1111;

    // NOTE: may be incorrect, but `FreelistDoubleBuffer` aligns all sizes
    type ArrayPadding = [u8; 0];
}

pub struct StaticObjectsIter<'a, A: MaterialArray<VertexAttributeKind>> {
    inner: std::slice::Iter<'a, SlotData<A>>,
    buffer_handle: StorageBufferHandle,
    slot: u32,
    len: u32,
}

impl<'a, A> StaticObjectsIter<'a, A>
where
    A: MaterialArray<VertexAttributeKind>,
{
    pub fn buffer_handle(&self) -> StorageBufferHandle {
        self.buffer_handle
    }
}

impl<'a, A> Iterator for StaticObjectsIter<'a, A>
where
    A: MaterialArray<VertexAttributeKind>,
{
    type Item = (
        u32,
        &'a GpuObject<<A as MaterialArray<VertexAttributeKind>>::U32Array>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.inner.next()? {
                Some(item) if item.enabled_object_data.is_some() => {
                    let slot = self.slot;
                    self.slot += 1;
                    break Some((slot, item));
                }
                _ => self.slot += 1,
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len as usize, Some(self.len as usize))
    }
}

impl<'a, A> ExactSizeIterator for StaticObjectsIter<'a, A> where
    A: MaterialArray<VertexAttributeKind>
{
}

pub(crate) struct WriteStaticObject<'a> {
    mesh: &'a GpuMesh,
    handle: RawStaticObjectHandle,
    object: Box<StaticObject>,
    object_manager: Option<&'a mut ObjectManager>,
}

impl WriteStaticObject<'_> {
    pub fn run<M: Material>(mut self, material_slot: u32) {
        let object_manager = self.object_manager.take().expect("must always be some");
        let archetype = object_manager.get_or_create_archetype::<M>();
        let handle = self.handle;

        let slot = self.fill_slot(
            material_slot,
            M::required_attributes().as_ref(),
            &M::supported_attributes(),
            archetype,
        );

        object_manager.static_handles.insert(
            handle,
            HandleData {
                archetype: TypeId::of::<M>(),
                slot,
            },
        );
    }

    fn fill_slot<A>(
        self,
        material_slot: u32,
        required_attributes: &[VertexAttributeKind],
        supported_attributes: &A,
        archetype: &mut ObjectArchetype,
    ) -> u32
    where
        A: MaterialArray<VertexAttributeKind>,
    {
        let required_attributes_mask = required_attributes
            .iter()
            .fold(0u8, |mask, attribute| mask | *attribute as u8);
        let mesh_attributes_mask = self
            .mesh
            .attributes()
            .fold(0u8, |mask, attribute| mask | attribute as u8);

        assert_eq!(
            mesh_attributes_mask & required_attributes_mask,
            required_attributes_mask
        );

        let vertex_attribute_offsets = supported_attributes.clone().map_to_u32(|attribute| {
            match self.mesh.get_attribute_range(attribute) {
                Some(range) => range.start,
                None => u32::MAX,
            }
        });

        let indices = self.mesh.indices();
        let first_index = indices.start;
        let index_count = indices.end - indices.start;

        // Compute bounding sphere in global space
        let mesh_bounding_sphere = *self.mesh.bounding_sphere();
        let global_bounding_sphere = mesh_bounding_sphere.transformed(&self.object.transform);

        let gpu_object = GpuObject::<A::U32Array> {
            enabled_object_data: Some(EnabledObjectData {
                mesh_handle: self.object.mesh,
                material_handle: self.object.material,
            }),
            mesh_bounding_sphere,
            transform: self.object.transform,
            global_bounding_sphere,
            vertex_attribute_offsets,
            first_index,
            index_count,
            material_slot,
        };

        let slot = archetype.free_slots.pop().unwrap_or_else(|| {
            let slot = archetype.next_slot;
            archetype.next_slot += 1;
            slot
        });

        {
            // SAFETY: `downcast_mut` template parameter is the same as the one used to
            // construct `archetype`. (material -> explicit attributes)
            let mut data = unsafe { archetype.data.downcast_mut::<SlotData<A>>() };
            if slot as usize >= data.len() {
                let size = slot.checked_next_power_of_two().expect("too many slots");
                data.resize_with(size as usize + 1, || None);
            }
            data[slot as usize] = Some(gpu_object);
        }

        archetype.buffer.update_slot(slot);
        archetype.active_object_count += 1;
        slot
    }
}

struct FlushObject<'a> {
    device: &'a gfx::Device,
    encoder: &'a mut gfx::Encoder,
    scatter_copy: &'a ScatterCopy,
    bindless_resources: &'a BindlessResources,
}

fn flush<A: MaterialArray<VertexAttributeKind>>(
    archetype: &mut ObjectArchetype,
    args: FlushObject,
) -> Result<()> {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to
    // construct `archetype`.
    let data = unsafe { archetype.data.typed_data::<SlotData<A>>() };

    // SAFETY: `flush` is called with the same template parameter all the time.
    unsafe {
        archetype
            .buffer
            .flush::<<GpuObject<A::U32Array> as gfx::AsStd430>::Output, _>(
                args.device,
                args.encoder,
                args.scatter_copy,
                args.bindless_resources,
                |slot| {
                    let material = data[slot as usize].as_ref().expect("invalid slot");
                    material.as_std430()
                },
            )?;
    }

    Ok(())
}

fn update_transform<A: MaterialArray<VertexAttributeKind>>(
    archetype: &mut ObjectArchetype,
    slot: u32,
    transform: &Mat4,
) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to
    // construct `data`.
    let mut data = unsafe { archetype.data.downcast_mut::<SlotData<A>>() };
    let item = data.get_mut(slot as usize).expect("invalid handle slot");
    let item = item.as_mut().expect("value was not initialized");

    item.transform = *transform;
    item.global_bounding_sphere = item.mesh_bounding_sphere.transformed(transform);

    archetype.buffer.update_slot(slot);
}

fn remove<A: MaterialArray<VertexAttributeKind>>(archetype: &mut ObjectArchetype, slot: u32) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to
    // construct `data`.
    let mut data = unsafe { archetype.data.downcast_mut::<SlotData<A>>() };
    let item = data.get_mut(slot as usize).expect("invalid handle slot");
    let item = item.as_mut().expect("value was not initialized");

    // Set item as disabled and mark it as updated to flush the data to the GPU.
    item.enabled_object_data = None;
    archetype.buffer.update_slot(slot);

    // It is ok to add this slot to available, since an inserted object will
    // overwrite the stored value (including the `enabled`) during this frame,
    // and this slot was already marked as updated.
    archetype.free_slots.push(slot);
    archetype.active_object_count -= 1;
}
