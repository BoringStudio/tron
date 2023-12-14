use std::collections::{HashMap, HashSet};

pub mod util;

pub type FastHashSet<K> = HashSet<K, ahash::RandomState>;
pub type FastHashMap<K, V> = HashMap<K, V, ahash::RandomState>;
