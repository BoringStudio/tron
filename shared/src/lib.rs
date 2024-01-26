use std::collections::{HashMap, HashSet};

pub use self::any_vec::AnyVec;

pub mod any_vec;
pub mod hlist;
pub mod util;

pub type FastHashSet<K> = HashSet<K, ahash::RandomState>;
pub type FastHashMap<K, V> = HashMap<K, V, ahash::RandomState>;

pub type FastDashSet<K> = dashmap::DashSet<K, ahash::RandomState>;
pub type FastDashMap<K, V> = dashmap::DashMap<K, V, ahash::RandomState>;

pub trait Embed {
    fn iter() -> impl Iterator<Item = (&'static str, &'static [u8])>;
}

#[macro_export]
macro_rules! embed {
    (
        $(#[$meta:meta])*
        $vis:vis $ident:ident($base:literal) = [$($item:literal),*$(,)?]
    ) => {
        $(#[$meta])*
        $vis struct $ident;

        impl $crate::Embed for $ident {
            fn iter() -> impl Iterator<Item = (&'static str, &'static [u8])> {
                [$(($item, include_bytes!(concat!($base, "/", $item)).as_ref())),*].into_iter()
            }
        }
    };
}
