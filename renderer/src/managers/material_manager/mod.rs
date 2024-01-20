use std::any::TypeId;
use std::collections::hash_map;

use anyhow::Result;
use shared::FastHashMap;

use self::material_buffer::MaterialBuffer;
use self::material_data::MaterialData;
use crate::managers::GpuMesh;
use crate::types::{Material, MaterialArray, RawMaterialHandle};
use crate::util::ScatterCopy;

mod material_buffer;
mod material_data;

#[derive(Default)]
pub struct MaterialManager {
    handles: FastHashMap<RawMaterialHandle, HandleData>,
    archetypes: FastHashMap<TypeId, MaterialArchetype>,
}

impl MaterialManager {
    #[tracing::instrument(level = "debug", name = "insert_material", skip_all)]
    pub fn insert<M: Material>(
        &mut self,
        device: &gfx::Device,
        handle: RawMaterialHandle,
        material: M,
    ) -> Result<(), gfx::OutOfDeviceMemory> {
        let archetype = self.get_or_create_archetype::<M>(device)?;

        let slot = archetype.free_slots.pop().unwrap_or_else(|| {
            let slot = archetype.next_slot;
            archetype.next_slot += 1;
            slot
        });

        {
            // SAFETY: `downcast_mut` template parameter is the same as the one used to
            // construct `archetype`.
            let mut data = unsafe { archetype.data.downcast_mut::<Option<M>>() };
            if slot >= data.len() {
                data.resize_with(slot.next_power_of_two(), || None);
            }
            data[slot] = Some(material);
        }

        archetype.buffer.update_slot(slot);
        self.handles.insert(
            handle,
            HandleData {
                archetype: TypeId::of::<M>(),
                slot,
            },
        );

        Ok(())
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
        let item = data.get_mut(*slot).expect("invalid handle slot");
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

        (archetype.remove_data)(&mut archetype.data, *slot);
        archetype.free_slots.push(*slot);
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
                    let material = data[slot].as_ref().expect("invalid slot");
                    material.shader_data()
                },
            )?;
        }

        Ok(())
    }

    fn get_or_create_archetype<M: Material>(
        &mut self,
        device: &gfx::Device,
    ) -> Result<&mut MaterialArchetype, gfx::OutOfDeviceMemory> {
        let id = TypeId::of::<M>();
        match self.archetypes.entry(id) {
            hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
            hash_map::Entry::Vacant(entry) => {
                let buffer = MaterialBuffer::new::<M::ShaderDataType>(device)?;
                Ok(entry.insert(MaterialArchetype {
                    data: MaterialData::new::<Option<M>>(),
                    buffer,
                    next_slot: 0,
                    free_slots: Vec::new(),
                    remove_data: remove_data::<M>,
                }))
            }
        }
    }
}

struct HandleData {
    archetype: TypeId,
    slot: usize,
}

struct MaterialArchetype {
    data: MaterialData,
    buffer: MaterialBuffer,
    next_slot: usize,
    free_slots: Vec<usize>,
    remove_data: fn(&mut MaterialData, usize),
}

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
            Some(range) => range.start,
            None => u32::MAX,
        }
    });

    let indices = args.mesh.indices();
    let first_index = indices.start;
    let index_count = indices.end - indices.start;

    // TODO: add object
}

fn remove_data<M: Material>(data: &mut MaterialData, slot: usize) {
    // SAFETY: `downcast_mut` template parameter is the same as the one used to
    // construct `data`.
    let mut data = unsafe { data.downcast_mut::<Option<M>>() };
    let item = data.get_mut(slot).expect("invalid handle slot");
    std::mem::take(item).expect("value was not initialized");
}
