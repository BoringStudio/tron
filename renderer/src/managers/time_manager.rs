use std::time::{Duration, Instant};

#[derive(Default)]
pub struct TimeManager {
    fixed_update: Option<FixedUpdateInfo>,
}

impl TimeManager {
    pub fn updated_fixed_time(&mut self, updated_at: Instant, duration: Duration) {
        let duration_sec = duration.as_secs_f64();
        self.fixed_update = (duration_sec > MIN_FRAME_DURATION).then_some(FixedUpdateInfo {
            updated_at,
            prev_interval_sec: duration_sec,
        });
    }

    pub fn compute_interpolation_factor(&self, rendered_at: Instant) -> f32 {
        let Some(state) = &self.fixed_update else {
            return 1.0;
        };

        // TODO: add noise filter?
        let since_fixed_update = rendered_at.duration_since(state.updated_at).as_secs_f64();
        (since_fixed_update / state.prev_interval_sec) as f32
    }
}

struct FixedUpdateInfo {
    updated_at: Instant,
    prev_interval_sec: f64,
}

const MIN_FRAME_DURATION: f64 = 0.000001;
