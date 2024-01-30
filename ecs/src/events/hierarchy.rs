use bevy_ecs::entity::Entity;
use bevy_ecs::event::Event;

#[derive(Debug, Clone, PartialEq, Eq, Event)]
pub enum HierarchyEvent {
    ChildAdded {
        child: Entity,
        parent: Entity,
    },
    ChildRemoved {
        child: Entity,
        parent: Entity,
    },
    ChildMoved {
        child: Entity,
        old_parent: Entity,
        new_parent: Entity,
    },
}
