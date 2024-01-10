use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

pub trait VertexAttribute: std::fmt::Debug + Default + PartialEq + Pod + Send + Sync {
    const FORMAT: gfx::VertexFormat;
    const KIND: VertexAttributeKind;
}

macro_rules! define_vertex_attributes {
    (
        $(#[$kind_meta:meta])* kind: $kind:ident;
        $($(#[$ident_meta:meta])* $ident:ident($inner:ty) => $format:ident;)*
    ) => {
        $(#[$kind_meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $kind {
            $($ident,)*
        }

        $(
            $(#[$ident_meta])*
            #[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
            #[repr(transparent)]
            pub struct $ident(pub $inner);

            impl VertexAttribute for $ident {
                const FORMAT: gfx::VertexFormat = gfx::VertexFormat::$format;
                const KIND: VertexAttributeKind = VertexAttributeKind::$ident;
            }

            impl From<$inner> for $ident {
                #[inline]
                fn from(value: $inner) -> Self {
                    Self(value)
                }
            }

            impl From<$ident> for $inner {
                #[inline]
                fn from($ident(value): $ident) -> Self {
                    value
                }
            }

            impl AsRef<$inner> for $ident {
                #[inline]
                fn as_ref(&self) -> &$inner {
                    &self.0
                }
            }

            impl AsMut<$inner> for $ident {
                #[inline]
                fn as_mut(&mut self) -> &mut $inner {
                    &mut self.0
                }
            }

            impl std::ops::Deref for $ident {
                type Target = $inner;

                #[inline]
                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl std::ops::DerefMut for $ident {
                #[inline]
                fn deref_mut(&mut self) -> &mut Self::Target {
                    &mut self.0
                }
            }
        )*
    };
}

define_vertex_attributes! {
    /// The kind of a vertex attribute.
    kind: VertexAttributeKind;

    /// A 2D position.
    Position2(Vec2) => Float32x2;
    /// A 3D position.
    Position3(Vec3) => Float32x3;
    /// A normal vector.
    Normal(Vec3) => Float32x3;
    /// A tangent vector.
    Tangent(Vec3) => Float32x3;
    /// A local UV coordinate.
    UV0(Vec2) => Float32x2;
    /// RGBA color.
    Color(Vec4) => Float32x4;
}

pub struct VertexAttributeData {
    kind: VertexAttributeKind,
    ptr: *mut u8,
    byte_len: usize,
    drop_fn: unsafe fn(*mut u8, usize),
}

impl VertexAttributeData {
    pub fn new<T: VertexAttribute>(mut data: Vec<T>) -> Self {
        assert!(std::mem::size_of::<T>() != 0);

        // Ensure that capacity is equal to len.
        data.shrink_to_fit();
        debug_assert!(data.len() == data.capacity());

        let mut data = std::mem::ManuallyDrop::new(data);
        let ptr = data.as_mut_ptr();
        let bytes = std::mem::size_of_val::<[T]>(data.as_slice());

        Self {
            kind: T::KIND,
            ptr: ptr.cast(),
            byte_len: bytes,
            drop_fn: drop_vec::<T>,
        }
    }

    pub fn kind(&self) -> VertexAttributeKind {
        self.kind
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn untyped_data(&self) -> &[u8] {
        // SAFETY: `self.ptr` is a valid pointer to a slice of `self.byte_len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.byte_len) }
    }

    pub fn typed_data<T: VertexAttribute>(&self) -> Option<&[T]> {
        if self.kind == T::KIND {
            Some(bytemuck::cast_slice(self.untyped_data()))
        } else {
            None
        }
    }

    pub fn typed_data_mut<T: VertexAttribute>(&mut self) -> Option<&mut [T]> {
        if self.kind == T::KIND {
            // SAFETY: `self.ptr` is a valid pointer to a slice of `self.byte_len` bytes.
            let data = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.byte_len) };
            Some(bytemuck::cast_slice_mut(data))
        } else {
            None
        }
    }
}

impl<T: VertexAttribute> From<Vec<T>> for VertexAttributeData {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}

impl Drop for VertexAttributeData {
    fn drop(&mut self) {
        // SAFETY:
        // - `T` is not a ZST.
        // - `self.ptr` was aquired from a `Vec<T>` with a length equal to capacity.
        // - `self.byte_len` is equal to `vec.len() * std::mem::size_of::<T>()`.
        unsafe { (self.drop_fn)(self.ptr, self.byte_len) }
    }
}

// SAFETY: `VertexAttributeData` can only be constructed from `Vec<T>`
// where `T: VertexAttribute`, where `VertexAttribute: Send + Sync`.
unsafe impl Send for VertexAttributeData {}
unsafe impl Sync for VertexAttributeData {}

/// # Safety
/// The following must be true:
/// - `T` must not be a ZST.
/// - `ptr` must be aquired from a `Vec<T>` with a length equal to capacity.
/// - `bytes` must be equal to `vec.len() * std::mem::size_of::<T>()`.
unsafe fn drop_vec<T>(ptr: *mut u8, bytes: usize) {
    let len = bytes / std::mem::size_of::<T>();
    Vec::<T>::from_raw_parts(ptr.cast(), len, len);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_with_different_alignment() {
        const POSITIONS: &[Position2] = &[
            Position2(Vec2::new(1.0, 2.0)),
            Position2(Vec2::new(3.0, 4.0)),
        ];

        let positions = POSITIONS.to_owned();
        let mut attribute = VertexAttributeData::new(positions);
        assert_eq!(attribute.byte_len(), 16);
        assert_eq!(attribute.untyped_data().len(), 16);
        assert_eq!(attribute.typed_data::<Position2>(), Some(POSITIONS));
        assert_eq!(attribute.typed_data::<UV0>(), None);

        assert_eq!(
            attribute.typed_data_mut::<Position2>(),
            Some(&mut [
                Position2(Vec2::new(1.0, 2.0)),
                Position2(Vec2::new(3.0, 4.0)),
            ] as &mut [Position2])
        );
        assert_eq!(attribute.typed_data_mut::<UV0>(), None);
    }

    #[test]
    fn from_vec_with_extra_capacity() {
        const POSITIONS: &[Position2] = &[
            Position2(Vec2::new(1.0, 2.0)),
            Position2(Vec2::new(3.0, 4.0)),
        ];

        let mut positions = POSITIONS.to_owned();

        positions.reserve(10);
        assert!(positions.capacity() > positions.len());

        let mut attribute = VertexAttributeData::new(positions);
        assert_eq!(attribute.byte_len(), 16);
        assert_eq!(attribute.untyped_data().len(), 16);
        assert_eq!(attribute.typed_data::<Position2>(), Some(POSITIONS));
        assert_eq!(attribute.typed_data::<UV0>(), None);

        assert_eq!(
            attribute.typed_data_mut::<Position2>(),
            Some(&mut [
                Position2(Vec2::new(1.0, 2.0)),
                Position2(Vec2::new(3.0, 4.0)),
            ] as &mut [Position2])
        );
        assert_eq!(attribute.typed_data_mut::<UV0>(), None);
    }
}
