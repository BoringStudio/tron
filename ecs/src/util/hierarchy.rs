use bevy_ecs::bundle::Bundle;
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::World;
use bevy_ecs::system::{Command, Commands, EntityCommands};
use smallvec::SmallVec;

pub struct ChildBuilder<'a> {
    commands: Commands<'a, 'a>,
    push_children: PushChildren,
}

impl ChildBuilder<'_> {
    pub fn spawn(&mut self, bundle: impl Bundle) -> EntityCommands {
        let entity = self.commands.spawn(bundle);
        self.push_children.children.push(entity.id());
        entity
    }

    pub fn spawn_empty(&mut self) -> EntityCommands {
        let entity = self.commands.spawn_empty();
        self.push_children.children.push(entity.id());
        entity
    }

    pub fn parent(&self) -> Entity {
        self.push_children.parent
    }

    pub fn add_command(&mut self, command: impl Command + 'static) -> &mut Self {
        self.commands.add(command);
        self
    }
}

trait BuildChildren {
    fn with_children(&mut self, spawn_children: impl FnOnce(&mut ChildBuilder)) -> &mut Self;

    fn push_children(&mut self, children: &[Entity]) -> &mut Self;

    fn insert_children(&mut self, index: usize, children: &[Entity]) -> &mut Self;

    fn add_child(&mut self, child: Entity) -> &mut Self;

    fn clear_children(&mut self) -> &mut Self;

    fn replace_children(&mut self) -> &mut Self;

    fn set_parent(&mut self, parent: Entity) -> &mut Self;

    fn remove_parent(&mut self) -> &mut Self;
}

impl<'a> BuildChildren for EntityCommands<'a> {
    fn with_children(&mut self, spawn_children: impl FnOnce(&mut ChildBuilder)) -> &mut Self {
        let parent = self.id();
        let mut builder = ChildBuilder {
            commands: self.commands(),
            push_children: PushChildren {
                parent,
                children: Default::default(),
            },
        };
        spawn_children(&mut builder);

        let children = builder.push_children;
        if children.children.contains(&parent) {
            panic!("entity cannot be child of itself");
        }

        self.commands().add(children);
        self
    }

    fn push_children(&mut self, children: &[Entity]) -> &mut Self {
        let parent = self.id();
        if children.contains(&parent) {
            panic!("entity cannot be child of itself");
        }
        self.commands().add(PushChildren {
            children: SmallVec::from(children),
            parent,
        });
        self
    }

    fn insert_children(&mut self, index: usize, children: &[Entity]) -> &mut Self {
        todo!()
    }

    fn add_child(&mut self, child: Entity) -> &mut Self {
        todo!()
    }

    fn clear_children(&mut self) -> &mut Self {
        todo!()
    }

    fn replace_children(&mut self) -> &mut Self {
        todo!()
    }

    fn set_parent(&mut self, parent: Entity) -> &mut Self {
        todo!()
    }

    fn remove_parent(&mut self) -> &mut Self {
        todo!()
    }
}

/// A command with adds children to the parent.
pub struct PushChildren {
    parent: Entity,
    children: SmallVec<[Entity; 8]>,
}

impl Command for PushChildren {
    fn apply(self, world: &mut World) {
        todo!()
    }
}
