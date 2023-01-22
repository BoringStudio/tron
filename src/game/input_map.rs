use std::hash::Hash;

use anyhow::Result;
use winit::event::VirtualKeyCode;

pub type KeyCode = VirtualKeyCode;

pub struct InputMap<T> {
    keys_to_actions: nohash_hasher::IntMap<u32, T>,
    actions_to_keys: nohash_hasher::IntMap<T, u32>,
    pressed_keys: nohash_hasher::IntSet<u32>,
}

impl<T> Default for InputMap<T> {
    fn default() -> Self {
        Self {
            keys_to_actions: Default::default(),
            actions_to_keys: Default::default(),
            pressed_keys: Default::default(),
        }
    }
}

impl<T> InputMap<T>
where
    T: Hash + Eq + Copy + std::fmt::Debug + nohash_hasher::IsEnabled,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_action_key(&mut self, action: T, key_code: KeyCode) -> Result<()> {
        if let Some(old_key) = self.actions_to_keys.insert(action, key_code as u32) {
            self.keys_to_actions.remove(&old_key);
        }
        if let Some(old_action) = self.keys_to_actions.insert(key_code as u32, action) {
            anyhow::bail!("key {key_code:?} already used for {old_action:?}");
        }
        Ok(())
    }

    pub fn is_active(&self, action: T) -> bool {
        if let Some(key_code) = self.actions_to_keys.get(&action) {
            self.pressed_keys.contains(key_code)
        } else {
            false
        }
    }

    pub fn handle_input(&mut self, input: &winit::event::KeyboardInput) -> Option<InputState<T>> {
        use winit::event::ElementState;

        let key_code = input.virtual_keycode?;
        let action = *self.keys_to_actions.get(&(key_code as u32))?;

        Some(match input.state {
            ElementState::Pressed => InputState::Pressed(action),
            ElementState::Released => InputState::Released(action),
        })
    }
}

pub enum InputState<T> {
    Pressed(T),
    Released(T),
}

impl<T> InputState<T> {
    pub fn as_any(self) -> T {
        match self {
            Self::Pressed(action) | Self::Released(action) => action,
        }
    }
}
