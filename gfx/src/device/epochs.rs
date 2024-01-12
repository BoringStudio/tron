use std::collections::VecDeque;
use std::sync::Mutex;

use shared::FastHashMap;

use crate::encoder::CommandBuffer;
use crate::queue::QueueId;

pub(crate) struct Epochs {
    queues: FastHashMap<QueueId, Mutex<QueueEpochs>>,
}

impl Epochs {
    pub fn new(queues: impl IntoIterator<Item = QueueId>) -> Self {
        Self {
            queues: queues
                .into_iter()
                .map(|id| (id, Mutex::new(QueueEpochs::default())))
                .collect(),
        }
    }

    pub fn next_epoch(&self, queue: QueueId) -> u64 {
        self.queues[&queue].lock().unwrap().next_epoch()
    }

    pub fn next_epoch_all_queues(&self) -> Vec<(QueueId, u64)> {
        self.queues
            .iter()
            .map(|(&id, queue)| (id, queue.lock().unwrap().next_epoch()))
            .collect()
    }

    pub fn close_epoch(&self, queue: QueueId, epoch: u64) {
        self.queues[&queue].lock().unwrap().close_epoch(epoch);
    }

    pub fn drain_free_command_buffers(
        &self,
        queue: QueueId,
        primary: &mut Vec<CommandBuffer>,
        secondaty: &mut Vec<CommandBuffer>,
    ) {
        let mut queue = self.queues[&queue].lock().unwrap();
        primary.append(&mut queue.free_primary_buffers);
        secondaty.append(&mut queue.free_secondary_buffers);
    }

    pub fn submit(&self, queue: QueueId, command_buffers: impl Iterator<Item = CommandBuffer>) {
        let mut queue = self.queues[&queue].lock().unwrap();
        let epoch = queue.epochs.front_mut().unwrap();
        epoch.command_buffers.extend(command_buffers);
    }
}

#[derive(Default)]
struct QueueEpochs {
    next: u64,
    epochs: VecDeque<Epoch>,
    epochs_cache: VecDeque<Epoch>,
    free_primary_buffers: Vec<CommandBuffer>,
    free_secondary_buffers: Vec<CommandBuffer>,
}

impl QueueEpochs {
    fn next_epoch(&mut self) -> u64 {
        let new_epoch = self.epochs_cache.pop_front().unwrap_or_default();
        self.epochs.push_front(new_epoch);

        let current = self.next;
        self.next += 1;
        current
    }

    fn close_epoch(&mut self, epoch: u64) {
        debug_assert!(epoch < self.next);

        let n = (self.next - epoch) as usize;
        if n < self.epochs.len() {
            for mut epoch in self.epochs.drain(n..) {
                for mut command_buffer in epoch.command_buffers.drain(..) {
                    command_buffer.clear_references();
                    self.free_secondary_buffers
                        .extend(command_buffer.drain_secondary_buffers());
                    self.free_primary_buffers.push(command_buffer);
                }
                self.epochs_cache.push_back(epoch);
            }
        }
    }
}

impl Drop for QueueEpochs {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(
                self.free_primary_buffers
                    .iter()
                    .all(|cb| cb.references().is_empty()),
                "all free primary command buffers must be cleared"
            );
            assert!(
                self.free_secondary_buffers
                    .iter()
                    .all(|cb| cb.references().is_empty()),
                "all free secondary command buffers must be cleared"
            );
            assert!(
                self.epochs
                    .iter()
                    .all(|epoch| epoch.command_buffers.is_empty()),
                "all epochs must be flushed"
            )
        }
    }
}

#[derive(Default)]
struct Epoch {
    command_buffers: Vec<CommandBuffer>,
}
