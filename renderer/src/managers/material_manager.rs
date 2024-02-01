use std::any::TypeId;
use std::collections::hash_map;

use anyhow::Result;
use shared::any::AnyVec;
use shared::FastHashMap;

use crate::managers::object_manager::{WriteDynamicObject, WriteStaticObject};
use crate::types::{MaterialInstance, RawMaterialInstanceHandle};
use crate::util::{BindlessResources, FreelistDoubleBuffer, ScatterCopy, StorageBufferHandle};

#[derive(Default)]
pub struct MaterialManager {
    handles: FastHashMap<RawMaterialInstanceHandle, HandleData>,
    archetypes: FastHashMap<TypeId, MaterialArchetype>,
}

impl MaterialManager {
    pub fn materials_data_buffer_handle<M: MaterialInstance>(&self) -> Option<StorageBufferHandle> {
        let archetype = self.archetypes.get(&TypeId::of::<M>())?;
        Some(archetype.buffer.handle())
    }

    #[tracing::instrument(level = "debug", name = "insert_material", skip_all)]
    pub fn insert_material_instance<M: MaterialInstance>(
        &mut self,
        handle: RawMaterialInstanceHandle,
        material: M,
    ) {
        let archetype = self.get_or_create_archetype::<M>();

        let slot = archetype.free_slots.pop().unwrap_or_else(|| {
            let slot = archetype.next_slot;
            archetype.next_slot += 1;
            slot
        });

        {
            // SAFETY: `downcast_mut` template parameter is the same as the one used to
            // construct `archetype`.
            let mut data = unsafe { archetype.data.downcast_mut::<SlotData<M>>() };
            if slot as usize >= data.len() {
                let size = slot.checked_next_power_of_two().expect("too many slots");
                data.resize_with(size as usize + 1, || None);
            }
            data[slot as usize] = Some(material);
        }

        archetype.buffer.update_slot(slot);
        self.handles.insert(
            handle,
            HandleData {
                archetype: TypeId::of::<M>(),
                slot,
            },
        );
    }

    #[tracing::instrument(level = "debug", name = "update_material", skip_all)]
    pub fn update<M: MaterialInstance>(&mut self, handle: RawMaterialInstanceHandle, material: M) {
        let HandleData { archetype, slot } = &self.handles[&handle];
        assert_eq!(*archetype, TypeId::of::<M>());

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        // SAFETY: `typed_data_mut` template parameter is the same as the one used to
        // construct `archetype`.
        let mut data = unsafe { archetype.data.typed_data_mut::<SlotData<M>>() };
        let item = data.get_mut(*slot as usize).expect("invalid handle slot");
        *item.as_mut().expect("value was not initialized") = material;

        archetype.buffer.update_slot(*slot);
    }

    #[tracing::instrument(level = "debug", name = "remove_material", skip_all)]
    pub fn remove(&mut self, handle: RawMaterialInstanceHandle) {
        let HandleData { archetype, slot } = &self.handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.remove_slot)(archetype, *slot);
    }

    #[tracing::instrument(level = "debug", name = "flush_materials", skip_all)]
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
                FlushMaterial {
                    device,
                    encoder,
                    scatter_copy,
                    bindless_resources,
                },
            )?;
        }
        Ok(())
    }

    pub(crate) fn write_static_object(
        &mut self,
        handle: RawMaterialInstanceHandle,
        args: WriteStaticObject,
    ) {
        let HandleData { archetype, slot } = &self.handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.write_static_object)(archetype, *slot, args);
    }

    pub(crate) fn write_dynamic_object(
        &mut self,
        handle: RawMaterialInstanceHandle,
        args: WriteDynamicObject,
    ) {
        let HandleData { archetype, slot } = &self.handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.write_dynamic_object)(archetype, *slot, args);
    }

    fn get_or_create_archetype<M: MaterialInstance>(&mut self) -> &mut MaterialArchetype {
        let id = TypeId::of::<M>();
        match self.archetypes.entry(id) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => entry.insert(MaterialArchetype {
                data: AnyVec::new::<SlotData<M>>(),
                buffer: FreelistDoubleBuffer::with_capacity(INITIAL_BUFFER_CAPACITY),
                next_slot: 0,
                free_slots: Vec::new(),
                flush: flush::<M>,
                write_static_object: write_static_object::<M>,
                write_dynamic_object: write_dynamic_object::<M>,
                remove_slot: remove_slot::<M>,
            }),
        }
    }
}

const INITIAL_BUFFER_CAPACITY: u32 = 16;

struct HandleData {
    archetype: TypeId,
    slot: u32,
}

struct MaterialArchetype {
    data: AnyVec,
    buffer: FreelistDoubleBuffer,
    next_slot: u32,
    free_slots: Vec<u32>,
    flush: fn(&mut MaterialArchetype, FlushMaterial) -> Result<()>,
    write_static_object: fn(&MaterialArchetype, u32, WriteStaticObject),
    write_dynamic_object: fn(&MaterialArchetype, u32, WriteDynamicObject),
    remove_slot: fn(&mut MaterialArchetype, u32),
}

type SlotData<M> = Option<M>;

struct FlushMaterial<'a> {
    device: &'a gfx::Device,
    encoder: &'a mut gfx::Encoder,
    scatter_copy: &'a ScatterCopy,
    bindless_resources: &'a BindlessResources,
}

fn flush<M: MaterialInstance>(archetype: &mut MaterialArchetype, args: FlushMaterial) -> Result<()> {
    // SAFETY: `typed_data` template parameter is the same as the one used to
    // construct `archetype`.
    unsafe {
        let data = archetype.data.typed_data::<SlotData<M>>();
        archetype.buffer.flush::<M::ShaderDataType, _>(
            args.device,
            args.encoder,
            args.scatter_copy,
            args.bindless_resources,
            |slot| {
                let material = data[slot as usize].as_ref().expect("invalid slot");
                material.shader_data()
            },
        )?;
    }

    Ok(())
}

fn write_static_object<M: MaterialInstance>(
    _archetype: &MaterialArchetype,
    slot: u32,
    args: WriteStaticObject<'_>,
) {
    // NOTE: read material here if needed
    args.run::<M>(slot);
}

fn write_dynamic_object<M: MaterialInstance>(
    _archetype: &MaterialArchetype,
    slot: u32,
    args: WriteDynamicObject<'_>,
) {
    // NOTE: read material here if needed
    args.run::<M>(slot);
}

fn remove_slot<M: MaterialInstance>(archetype: &mut MaterialArchetype, slot: u32) {
    // SAFETY: `typed_data_mut` template parameter is the same as the one used to
    // construct `data`.
    let mut data = unsafe { archetype.data.typed_data_mut::<SlotData<M>>() };
    let item = data.get_mut(slot as usize).expect("invalid handle slot");
    std::mem::take(item).expect("value was not initialized");

    archetype.free_slots.push(slot);
}
