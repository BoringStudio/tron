#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceAddress(pub std::num::NonZeroU64);

impl DeviceAddress {
    pub fn new(address: u64) -> Option<Self> {
        std::num::NonZeroU64::new(address).map(Self)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum State<T> {
    Static(T),
    Dynamic,
}
