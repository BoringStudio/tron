/// Buffer device address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceAddress(pub std::num::NonZeroU64);

impl DeviceAddress {
    pub fn new(address: u64) -> Option<Self> {
        std::num::NonZeroU64::new(address).map(Self)
    }
}

/// Pipeline value state.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum State<T> {
    Static(T),
    Dynamic,
}

impl<T> State<T> {
    #[inline]
    pub const fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic)
    }
}
