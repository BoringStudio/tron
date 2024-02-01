use std::any::TypeId;
use std::collections::hash_map;

use anyhow::Result;
use gfx::AsStd430;
use glam::{Mat4, Quat, UVec4, Vec3, Vec4};
use shared::any::{AnyVec, AnyVecGuard};
use shared::packed::U32WithBool;
use shared::FastHashMap;

use crate::managers::{GpuMesh, MaterialManager, MeshManagerDataGuard};
use crate::types::{
    Material, MaterialArray, MaterialHandle, MeshHandle, ObjectData, RawDynamicObjectHandle,
    RawStaticObjectHandle, VertexAttributeKind,
};
use crate::util::{
    BindlessResources, BoundingSphere, FreelistDoubleBuffer, ScatterCopy, StorageBufferHandle,
};

#[derive(Default)]
pub struct ObjectManager {
    static_handles: FastHashMap<RawStaticObjectHandle, HandleData>,
    static_archetypes: FastHashMap<TypeId, StaticObjectArchetype>,
    dynamic_handles: FastHashMap<RawDynamicObjectHandle, HandleData>,
    dynamic_archetypes: FastHashMap<TypeId, DynamicObjectArchetype>,
}

impl ObjectManager {
    pub fn iter_static<M: Material>(
        &self,
    ) -> Option<StaticObjectsIter<'_, M::SupportedAttributes>> {
        let archetype = self.static_archetypes.get(&TypeId::of::<M>())?;

        // SAFETY: `typed_data` template parameter is the same as the one used to
        // construct `archetype`.
        let data = unsafe {
            archetype
                .data
                .typed_data::<StaticSlotData<M::SupportedAttributes>>()
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
        object: Box<ObjectData>,
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
            .static_archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.update_transform)(archetype, *slot, transform);
    }

    #[tracing::instrument(level = "debug", name = "remove_object", skip_all)]
    pub fn remove_static(&mut self, handle: RawStaticObjectHandle) {
        let HandleData { archetype, slot } = &self.static_handles[&handle];

        let archetype = self
            .static_archetypes
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
        for archetype in self.static_archetypes.values_mut() {
            (archetype.flush)(
                archetype,
                FlushStaticObject {
                    device,
                    encoder,
                    scatter_copy,
                    bindless_resources,
                },
            )?;
        }
        Ok(())
    }

    fn get_or_create_static_archetype<M: Material>(&mut self) -> &mut StaticObjectArchetype {
        let id = TypeId::of::<M>();
        match self.static_archetypes.entry(id) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => entry.insert(StaticObjectArchetype {
                data: AnyVec::new::<StaticSlotData<M::SupportedAttributes>>(),
                buffer: FreelistDoubleBuffer::with_capacity(INITIAL_BUFFER_CAPACITY),
                active_object_count: 0,
                next_slot: 0,
                free_slots: Vec::new(),
                flush: flush_static_object::<M::SupportedAttributes>,
                update_transform: update_static_object_transform::<M::SupportedAttributes>,
                remove: remove_static_object::<M::SupportedAttributes>,
            }),
        }
    }

    fn get_or_create_dynamic_archetype<M: Material>(&mut self) -> &mut DynamicObjectArchetype {
        let id = TypeId::of::<M>();
        match self.dynamic_archetypes.entry(id) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => entry.insert(DynamicObjectArchetype {
                data: AnyVec::new::<DynamicSlotData<M::SupportedAttributes>>(),
                active_object_count: 0,
                next_slot: 0,
                free_slots: Vec::new(),
                update_transform: update_dynamic_object_transform::<M::SupportedAttributes>,
                remove: remove_dynamic_object::<M::SupportedAttributes>,
            }),
        }
    }
}

const INITIAL_BUFFER_CAPACITY: u32 = 16;

struct HandleData {
    archetype: TypeId,
    slot: u32,
}

struct StaticObjectArchetype {
    data: AnyVec,
    buffer: FreelistDoubleBuffer,
    active_object_count: u32,
    next_slot: u32,
    free_slots: Vec<u32>,
    flush: fn(&mut StaticObjectArchetype, FlushStaticObject) -> Result<()>,
    update_transform: fn(&mut StaticObjectArchetype, u32, &Mat4),
    remove: fn(&mut StaticObjectArchetype, u32),
}

struct DynamicObjectArchetype {
    data: AnyVec,
    active_object_count: u32,
    next_slot: u32,
    free_slots: Vec<u32>,
    update_transform: fn(&mut DynamicObjectArchetype, u32, &Mat4, bool),
    remove: fn(&mut DynamicObjectArchetype, u32),
}

type StaticSlotData<A> =
    Option<GpuStaticObject<<A as MaterialArray<VertexAttributeKind>>::U32Array>>;
type DynamicSlotData<A> =
    Option<GpuDynamicObject<<A as MaterialArray<VertexAttributeKind>>::U32Array>>;

pub struct GpuStaticObject<A> {
    // NOTE: having `Some` here means that the object is enabled.
    // This is used to drop handles when the object is removed,
    // but allows to sync the GPU data with `enabled: false`.
    pub enabled_object_data: Option<EnabledObjectData>,
    pub mesh_bounding_sphere: BoundingSphere,

    pub global_transform: Mat4,
    pub global_bounding_sphere: BoundingSphere,
    pub vertex_attribute_offsets: A,
    pub first_index: u32,
    pub index_count: u32,
    pub material_slot: u32,
}

impl<A> GpuStaticObject<A> {
    fn make_data(&self) -> UVec4 {
        glam::uvec4(
            self.first_index,
            self.index_count,
            self.material_slot,
            self.enabled_object_data.is_some() as _,
        )
    }
}

impl<A> gfx::AsStd430 for GpuStaticObject<A>
where
    A: gfx::Std430,
{
    type Output = Std430GpuObject<A>;

    fn as_std430(&self) -> Self::Output {
        Std430GpuObject {
            transform: self.global_transform,
            transform_inverse_transpose: self.global_transform.inverse().transpose(),
            bounding_sphere: self.global_bounding_sphere.into(),
            data: self.make_data(),
            vertex_attribute_offsets: self.vertex_attribute_offsets,
        }
    }

    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst.transform = self.global_transform;
        dst.transform_inverse_transpose = self.global_transform.inverse().transpose();
        dst.bounding_sphere = self.global_bounding_sphere.into();
        dst.data = self.make_data();
        dst.vertex_attribute_offsets = self.vertex_attribute_offsets;
    }
}

pub struct GpuDynamicObject<A> {
    pub enabled_object_data: EnabledObjectData,
    pub mesh_bounding_sphere: BoundingSphere,

    pub prev_global_transform: GlobalTransform,
    pub next_global_transform: GlobalTransform,

    pub vertex_attribute_offsets: A,
    pub first_index: u32,
    // NOTE: `updated` flag is stored here to reduce the object size.
    // Index is unlikely to be greater than 2^31.
    pub index_count_and_updated: U32WithBool,
    pub material_slot: u32,
}

impl<A> GpuDynamicObject<A> {
    #[inline]
    pub fn is_updated(&self) -> bool {
        self.index_count_and_updated.get_bool()
    }

    #[inline]
    pub fn index_count(&self) -> u32 {
        self.index_count_and_updated.get_u32()
    }

    fn make_data(&self) -> UVec4 {
        glam::uvec4(
            self.first_index,
            self.index_count(),
            self.material_slot,
            true as _, // NOTE: dynamic objects are always enabled if they exist
        )
    }

    fn as_interpolated_std430(&self, t: f32) -> Std430GpuObject<A>
    where
        A: gfx::Std430,
    {
        let transform = if self.index_count_and_updated.get_bool() {
            self.prev_global_transform
                .to_interpolated_matrix(&self.next_global_transform, t)
        } else {
            // Skip interpolation if the object was not updated during the last frame.
            self.next_global_transform.to_matrix()
        };

        Std430GpuObject {
            transform_inverse_transpose: transform.inverse().transpose(),
            bounding_sphere: self.mesh_bounding_sphere.transformed(&transform).into(),
            transform,
            data: self.make_data(),
            vertex_attribute_offsets: self.vertex_attribute_offsets,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Std430GpuObject<A> {
    transform: Mat4,
    transform_inverse_transpose: Mat4,
    bounding_sphere: Vec4,
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

pub struct EnabledObjectData {
    pub _mesh_handle: MeshHandle,
    pub _material_handle: MaterialHandle,
}

#[derive(Clone, Copy)]
pub struct GlobalTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl GlobalTransform {
    fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    fn to_interpolated_matrix(&self, other: &Self, t: f32) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.scale.lerp(other.scale, t),
            self.rotation.slerp(other.rotation, t),
            self.translation.lerp(other.translation, t),
        )
    }
}

impl From<Mat4> for GlobalTransform {
    #[inline]
    fn from(matrix: Mat4) -> Self {
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
        Self {
            translation,
            rotation,
            scale,
        }
    }
}

pub struct StaticObjectsIter<'a, A: MaterialArray<VertexAttributeKind>> {
    inner: std::slice::Iter<'a, StaticSlotData<A>>,
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
        &'a GpuStaticObject<<A as MaterialArray<VertexAttributeKind>>::U32Array>,
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
    object: Box<ObjectData>,
    object_manager: Option<&'a mut ObjectManager>,
}

impl WriteStaticObject<'_> {
    pub fn run<M: Material>(mut self, material_slot: u32) {
        let object_manager = self.object_manager.take().expect("must always be some");
        let archetype = object_manager.get_or_create_static_archetype::<M>();
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
        archetype: &mut StaticObjectArchetype,
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
        let global_bounding_sphere =
            mesh_bounding_sphere.transformed(&self.object.global_transform);

        let gpu_object = GpuStaticObject::<A::U32Array> {
            enabled_object_data: Some(EnabledObjectData {
                _mesh_handle: self.object.mesh,
                _material_handle: self.object.material,
            }),
            mesh_bounding_sphere,
            global_transform: self.object.global_transform,
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
            let mut data = unsafe { archetype.data.downcast_mut::<StaticSlotData<A>>() };
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

struct FlushStaticObject<'a> {
    device: &'a gfx::Device,
    encoder: &'a mut gfx::Encoder,
    scatter_copy: &'a ScatterCopy,
    bindless_resources: &'a BindlessResources,
}

fn flush_static_object<A: MaterialArray<VertexAttributeKind>>(
    archetype: &mut StaticObjectArchetype,
    args: FlushStaticObject,
) -> Result<()> {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to
    // construct `archetype`.
    let data = unsafe { archetype.data.typed_data::<StaticSlotData<A>>() };

    // SAFETY: `flush` is called with the same template parameter all the time.
    unsafe {
        archetype
            .buffer
            .flush::<<GpuStaticObject<A::U32Array> as gfx::AsStd430>::Output, _>(
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

fn update_static_object_transform<A: MaterialArray<VertexAttributeKind>>(
    archetype: &mut StaticObjectArchetype,
    slot: u32,
    transform: &Mat4,
) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to construct `data`.
    let mut data = unsafe { archetype.data.downcast_mut::<StaticSlotData<A>>() };
    let item = expect_data_slot_mut(&mut data, slot);

    item.global_transform = *transform;
    item.global_bounding_sphere = item.mesh_bounding_sphere.transformed(transform);

    archetype.buffer.update_slot(slot);
}

fn update_dynamic_object_transform<A: MaterialArray<VertexAttributeKind>>(
    archetype: &mut DynamicObjectArchetype,
    slot: u32,
    transform: &Mat4,
    teleport: bool,
) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to construct `data`.
    let mut data = unsafe { archetype.data.downcast_mut::<DynamicSlotData<A>>() };
    let item = expect_data_slot_mut(&mut data, slot);

    // Update only the next transform. The previous will be updated
    // during the next flush...
    item.next_global_transform = GlobalTransform::from(*transform);
    if teleport {
        // ...unless the object was teleported, in which case we update
        // the previous transform as well.
        item.prev_global_transform = item.next_global_transform;
    }

    item.index_count_and_updated.set_bool(true);
}

fn remove_static_object<A: MaterialArray<VertexAttributeKind>>(
    archetype: &mut StaticObjectArchetype,
    slot: u32,
) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to construct `data`.
    let mut data = unsafe { archetype.data.downcast_mut::<StaticSlotData<A>>() };
    let item = expect_data_slot_mut(&mut data, slot);

    // Set item as disabled and mark it as updated to flush the data to the GPU.
    item.enabled_object_data = None;
    archetype.buffer.update_slot(slot);

    // It is ok to add this slot to available, since an inserted object will
    // overwrite the stored value (including the `enabled`) during this frame,
    // and this slot was already marked as updated.
    archetype.free_slots.push(slot);
    archetype.active_object_count -= 1;
}

fn remove_dynamic_object<A: MaterialArray<VertexAttributeKind>>(
    archetype: &mut DynamicObjectArchetype,
    slot: u32,
) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to construct `data`.
    let mut data = unsafe { archetype.data.downcast_mut::<DynamicSlotData<A>>() };
    let item = data.get_mut(slot as usize).expect("invalid handle slot");
    std::mem::take(item).expect("value was not initialized");

    archetype.free_slots.push(slot);
}

fn expect_data_slot_mut<'a, T: SlotDataExt>(
    data: &'a mut AnyVecGuard<'_, T>,
    slot: u32,
) -> &'a mut T::Inner {
    let item = data.get_mut(slot as usize).expect("invalid handle slot");
    item.as_mut().expect("value was not initialized")
}

trait SlotDataExt {
    type Inner;

    fn as_mut(&mut self) -> Option<&mut Self::Inner>;
}

impl<T> SlotDataExt for Option<T> {
    type Inner = T;

    #[inline]
    fn as_mut(&mut self) -> Option<&mut Self::Inner> {
        Option::as_mut(self)
    }
}
