use std::collections::{HashMap, HashSet};

pub mod util;

pub type FastHashSet<K> = HashSet<K, ahash::RandomState>;
pub type FastHashMap<K, V> = HashMap<K, V, ahash::RandomState>;

pub type FastDashSet<K> = dashmap::DashSet<K, ahash::RandomState>;
pub type FastDashMap<K, V> = dashmap::DashMap<K, V, ahash::RandomState>;
