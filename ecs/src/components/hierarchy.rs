use bevy_ecs::component::Component;
use bevy_ecs::entity::{Entity, EntityMapper, MapEntities};
use smallvec::SmallVec;

/// Holds a reference to the parent entity.
#[derive(Debug, PartialEq, Eq, Component)]
pub struct Parent(pub(crate) Entity);

impl Parent {
    #[inline]
    pub fn get(&self) -> Entity {
        self.0
    }
}

impl MapEntities for Parent {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        self.0 = entity_mapper.map_entity(self.0);
    }
}

impl std::ops::Deref for Parent {
    type Target = Entity;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Holds references to children entities.
#[derive(Debug, PartialEq, Eq, Component)]
pub struct Children(pub(crate) SmallVec<[Entity; 8]>);

impl Children {
    #[inline]
    pub fn swap(&mut self, index_a: usize, index_b: usize) {
        self.0.swap(index_a, index_b);
    }

    #[inline]
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&Entity, &Entity) -> std::cmp::Ordering,
    {
        self.0.sort_by(compare);
    }

    #[inline]
    pub fn sort_by_key<K, F>(&mut self, compare: F)
    where
        F: FnMut(&Entity) -> K,
        K: Ord,
    {
        self.0.sort_by_key(compare);
    }

    #[inline]
    pub fn sort_by_cached_key<K, F>(&mut self, compare: F)
    where
        F: FnMut(&Entity) -> K,
        K: Ord,
    {
        self.0.sort_by_cached_key(compare);
    }

    #[inline]
    pub fn sort_unstable_by<F>(&mut self, compare: F)
    where
        F: FnMut(&Entity, &Entity) -> std::cmp::Ordering,
    {
        self.0.sort_unstable_by(compare);
    }

    #[inline]
    pub fn sort_unstable_by_key<K, F>(&mut self, compare: F)
    where
        F: FnMut(&Entity) -> K,
        K: Ord,
    {
        self.0.sort_unstable_by_key(compare);
    }
}

impl MapEntities for Children {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        for child in self.0.iter_mut() {
            *child = entity_mapper.map_entity(*child);
        }
    }
}

impl std::ops::Deref for Children {
    type Target = [Entity];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl<'a> IntoIterator for &'a Children {
    type Item = &'a Entity;
    type IntoIter = std::slice::Iter<'a, Entity>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
