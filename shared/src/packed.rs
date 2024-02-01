#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct U32WithBool(pub u32);

impl U32WithBool {
    const BOOL_MASK: u32 = 0x8000_0000;

    #[inline]
    pub const fn new(value: u32, flag: bool) -> Self {
        debug_assert!(value & Self::BOOL_MASK == 0);
        Self(value | (flag as u32) << 31)
    }

    #[inline]
    pub fn set_u32(&mut self, value: u32) {
        debug_assert!(value & Self::BOOL_MASK == 0);
        self.0 = (self.0 & Self::BOOL_MASK) | value;
    }

    #[inline]
    pub const fn get_u32(&self) -> u32 {
        self.0 & !Self::BOOL_MASK
    }

    #[inline]
    pub fn set_bool(&mut self, value: bool) {
        self.0 = (self.0 & !Self::BOOL_MASK) | (value as u32) << 31;
    }

    #[inline]
    pub const fn get_bool(&self) -> bool {
        self.0 & Self::BOOL_MASK != 0
    }
}

impl From<u32> for U32WithBool {
    #[inline]
    fn from(value: u32) -> Self {
        Self::new(value, false)
    }
}

impl From<bool> for U32WithBool {
    #[inline]
    fn from(value: bool) -> Self {
        Self::new(0, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn still_u32() {
        assert_eq!(
            std::mem::align_of::<U32WithBool>(),
            std::mem::align_of::<u32>()
        );
        assert_eq!(
            std::mem::size_of::<U32WithBool>(),
            std::mem::size_of::<u32>()
        );
    }

    #[test]
    fn correct_init() {
        let empty = U32WithBool::default();
        assert_eq!(empty.get_u32(), 0);
        assert_eq!(empty.get_bool(), false);

        let max = U32WithBool::new(u32::MAX >> 1, true);
        assert_eq!(max.get_u32(), u32::MAX >> 1);
        assert_eq!(max.get_bool(), true);

        let only_bool = U32WithBool::new(0, true);
        assert_eq!(only_bool.get_u32(), 0);
        assert_eq!(only_bool.get_bool(), true);

        let only_u32 = U32WithBool::new(u32::MAX >> 1, false);
        assert_eq!(only_u32.get_u32(), u32::MAX >> 1);
        assert_eq!(only_u32.get_bool(), false);

        let some_u32_and_false = U32WithBool::new(123456, false);
        assert_eq!(some_u32_and_false.get_u32(), 123456);
        assert_eq!(some_u32_and_false.get_bool(), false);

        let some_u32_and_true = U32WithBool::new(123456, true);
        assert_eq!(some_u32_and_true.get_u32(), 123456);
        assert_eq!(some_u32_and_true.get_bool(), true);
    }

    #[test]
    fn correct_update() {
        let mut value = U32WithBool::default();
        assert_eq!(value.get_u32(), 0);
        assert_eq!(value.get_bool(), false);

        value.set_u32(123);
        assert_eq!(value.get_u32(), 123);
        assert_eq!(value.get_bool(), false);

        value.set_u32(u32::MAX >> 1);
        assert_eq!(value.get_u32(), u32::MAX >> 1);
        assert_eq!(value.get_bool(), false);

        value.set_bool(true);
        assert_eq!(value.get_u32(), u32::MAX >> 1);
        assert_eq!(value.get_bool(), true);

        value.set_u32(4123123);
        assert_eq!(value.get_u32(), 4123123);
        assert_eq!(value.get_bool(), true);
    }
}
