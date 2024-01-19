use crate::managers::GpuMesh;
use crate::types::{Material, MaterialArray, Sorting, VertexAttributeKind};

pub struct MaterialManager {}

struct MaterialArchetype {
    get_attributes: FnOnMaterialAttributes,
    record_object: FnAddObject,
}

type FnOnMaterialAttributes = fn(&mut dyn FnMut(&[VertexAttributeKind], &[VertexAttributeKind]));
type FnAddObject = fn(&[u8], AddObjectArgs<'_>);

struct AddObjectArgs<'a> {
    mesh: &'a GpuMesh,
}

fn record_object<M: Material>(_material: &M, args: AddObjectArgs<'_>) {
    let required_attributes_mask = M::required_attributes()
        .iter()
        .fold(0u8, |mask, attribute| mask | attribute as u8);
    let mesh_attributes_mask = args
        .mesh
        .attributes()
        .fold(0u8, |mask, attribute| mask | attribute as u8);

    assert!(mesh_attributes_mask & required_attributes_mask == required_attributes_mask);

    let vertex_attribute_offsets = M::supported_attributes().map_to_u32(|attribute| {
        match args.mesh.get_attribute_range(attribute) {
            Some(range) => range.start as u32,
            None => u32::MAX,
        }
    });

    let first_index = args.mesh.indices_range.start as u32;
    let index_count = (args.mesh.indices_range.end - args.mesh.indices_range.start) as u32;

    // TODO: add object
}
