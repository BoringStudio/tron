use std::any::TypeId;
use std::collections::hash_map;

use anyhow::Result;
use shared::{AnyVec, FastHashMap};

use crate::managers::object_manager::WriteStaticObject;
use crate::types::{Material, RawMaterialHandle};
use crate::util::{FreelistDoubleBuffer, ScatterCopy};

#[derive(Default)]
pub struct MaterialManager {
    handles: FastHashMap<RawMaterialHandle, HandleData>,
    archetypes: FastHashMap<TypeId, MaterialArchetype>,
}

impl MaterialManager {
    #[tracing::instrument(level = "debug", name = "insert_material", skip_all)]
    pub fn insert<M: Material>(&mut self, handle: RawMaterialHandle, material: M) {
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
                data.resize_with(size as usize, || None);
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
    pub fn update<M: Material>(&mut self, handle: RawMaterialHandle, material: M) {
        let HandleData { archetype, slot } = &self.handles[&handle];
        assert_eq!(*archetype, TypeId::of::<M>());

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        // SAFETY: `downcast_mut` template parameter is the same as the one used to
        // construct `archetype`.
        let mut data = unsafe { archetype.data.downcast_mut::<Option<M>>() };
        let item = data.get_mut(*slot as usize).expect("invalid handle slot");
        *item.as_mut().expect("value was not initialized") = material;

        archetype.buffer.update_slot(*slot);
    }

    #[tracing::instrument(level = "debug", name = "remove_material", skip_all)]
    pub fn remove(&mut self, handle: RawMaterialHandle) {
        let HandleData { archetype, slot } = &self.handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.remove)(archetype, *slot);
    }

    #[tracing::instrument(level = "debug", name = "flush_materials", skip_all)]
    pub fn flush<M: Material>(
        &mut self,
        device: &gfx::Device,
        encoder: &mut gfx::Encoder,
        scatter_copy: &ScatterCopy,
    ) -> Result<()> {
        let Some(archetype) = self.archetypes.get_mut(&TypeId::of::<M>()) else {
            return Ok(());
        };

        // SAFETY: `typed_data` template parameter is the same as the one used to
        // construct `archetype`.
        unsafe {
            let data = archetype.data.typed_data::<Option<M>>();
            archetype.buffer.flush::<M::ShaderDataType, _>(
                device,
                encoder,
                scatter_copy,
                |slot| {
                    let material = data[slot as usize].as_ref().expect("invalid slot");
                    material.shader_data()
                },
            )?;
        }

        Ok(())
    }

    pub(crate) fn write_static_object(
        &mut self,
        handle: RawMaterialHandle,
        args: WriteStaticObject,
    ) {
        let HandleData { archetype, slot } = &self.handles[&handle];

        let archetype = self
            .archetypes
            .get_mut(archetype)
            .expect("invalid handle archetype");

        (archetype.write_static_object)(archetype, *slot, args);
    }

    fn get_or_create_archetype<M: Material>(&mut self) -> &mut MaterialArchetype {
        let id = TypeId::of::<M>();
        match self.archetypes.entry(id) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => entry.insert(MaterialArchetype {
                data: AnyVec::new::<SlotData<M>>(),
                buffer: FreelistDoubleBuffer::with_capacity(INITIAL_BUFFER_CAPACITY),
                next_slot: 0,
                free_slots: Vec::new(),
                write_static_object: write_static_object::<M>,
                remove: remove::<M>,
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
    write_static_object: fn(&MaterialArchetype, u32, WriteStaticObject),
    remove: fn(&mut MaterialArchetype, u32),
}

type SlotData<M> = Option<M>;

fn write_static_object<M: Material>(
    _archetype: &MaterialArchetype,
    _slot: u32,
    args: WriteStaticObject,
) {
    // NOTE: read material here if needed
    args.run::<M>();
}

fn remove<M: Material>(archetype: &mut MaterialArchetype, slot: u32) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to
    // construct `data`.
    let mut data = unsafe { archetype.data.downcast_mut::<SlotData<M>>() };
    let item = data.get_mut(slot as usize).expect("invalid handle slot");
    std::mem::take(item).expect("value was not initialized");

    archetype.free_slots.push(slot);
}
