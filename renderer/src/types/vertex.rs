use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

pub trait PipelineVertexInputExt {
    fn bindings_and_attributes(
        &self,
    ) -> (Vec<gfx::VertexInputBinding>, Vec<gfx::VertexInputAttribute>);
}

impl<T: AsRef<[VertexLayout]>> PipelineVertexInputExt for T {
    fn bindings_and_attributes(
        &self,
    ) -> (Vec<gfx::VertexInputBinding>, Vec<gfx::VertexInputAttribute>) {
        fn bindings_and_attributes_impl(
            layouts: &[VertexLayout],
        ) -> (Vec<gfx::VertexInputBinding>, Vec<gfx::VertexInputAttribute>) {
            let mut bindings = Vec::with_capacity(layouts.len());
            let mut attributes = Vec::new();

            for (binding, layout) in layouts.iter().enumerate() {
                bindings.push(gfx::VertexInputBinding {
                    rate: layout.rate,
                    stride: layout.stride,
                });

                let base_location = attributes.len();
                attributes.extend(layout.locations.iter().enumerate().map(|(i, location)| {
                    gfx::VertexInputAttribute {
                        location: (base_location + i) as u32,
                        binding: binding as u32,
                        format: location.format,
                        offset: location.offset,
                    }
                }));
            }

            (bindings, attributes)
        }

        bindings_and_attributes_impl(self.as_ref())
    }
}

pub trait VertexType: std::fmt::Debug + Default + PartialEq + Pod {
    const LOCATIONS: &'static [VertexLocation];
    const RATE: gfx::VertexInputRate;

    fn layout() -> VertexLayout {
        VertexLayout {
            locations: Cow::Borrowed(Self::LOCATIONS),
            rate: Self::RATE,
            stride: std::mem::size_of::<Self>() as u32,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct VertexLayout {
    pub locations: Cow<'static, [VertexLocation]>,
    pub rate: gfx::VertexInputRate,
    pub stride: u32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VertexLocation {
    pub offset: u32,
    pub format: gfx::VertexFormat,
    pub component: VertexComponent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexComponent {
    Position2,
    Position3,
    Normal,
    Tangent,
    UV,
    Color,
}

pub trait VertexAttribute: std::fmt::Debug + Default + PartialEq + Pod {
    const FORMAT: gfx::VertexFormat;
    const COMPONENT: VertexComponent;
}

macro_rules! define_vertex_attributes {
    ($(
        $(#[$meta:meta])*
        $vis:vis $ident:ident($inner_vis:vis $inner:ty) => ($format:ident, $component:ident);
    )*) => {
        $(
            $(#[$meta])*
            $vis struct $ident($inner_vis $inner);

            impl VertexAttribute for $ident {
                const FORMAT: gfx::VertexFormat = gfx::VertexFormat::$format;
                const COMPONENT: VertexComponent = VertexComponent::$component;
            }
        )*
    };
}

define_vertex_attributes! {
    /// A 2D position.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
    #[repr(transparent)]
    pub Position2(pub Vec2) => (Float32x2, Position2);

    /// A 3D position.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
    #[repr(transparent)]
    pub Position3(pub Vec3) => (Float32x3, Position3);

    /// A normal vector.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
    #[repr(transparent)]
    pub Normal(pub Vec3) => (Float32x3, Normal);

    /// A tangent vector.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
    #[repr(transparent)]
    pub Tangent(pub Vec3) => (Float32x3, Tangent);

    /// A UV coordinate.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
    #[repr(transparent)]
    pub UV(pub Vec2) => (Float32x2, UV);

    /// RGBA color.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
    #[repr(transparent)]
    pub Color(pub Vec4) => (Float32x4, Color);
}

impl<T: VertexAttribute> VertexType for T {
    const LOCATIONS: &'static [VertexLocation] = &[VertexLocation {
        offset: 0,
        format: T::FORMAT,
        component: T::COMPONENT,
    }];
    const RATE: gfx::VertexInputRate = gfx::VertexInputRate::Vertex;
}

macro_rules! define_generic_vertex_types {
    ($(
        $(#[$meta:meta])*
        $vis:vis $ident:ident($($comp:tt),*);
    )*) => {$(
        $(#[$meta])*
        #[derive(Debug, Default, Clone, Copy, PartialEq)]
        #[repr(C)]
        pub struct $ident<$($comp),*>($(pub $comp),*);

        unsafe impl<$($comp: Pod),*> Pod for $ident<$($comp),*> {}
        unsafe impl<$($comp: Zeroable),*> Zeroable for $ident<$($comp),*> {}

        impl<$($comp),*> VertexType for $ident<$($comp),*>
        where
            $($comp: VertexAttribute),*
        {
            const LOCATIONS: &'static [VertexLocation] =
                &define_generic_vertex_types!(@location []; 0; $($comp),*);
            const RATE: gfx::VertexInputRate = gfx::VertexInputRate::Vertex;
        }
    )*};

    (@location [ $($t:expr,)* ]; $offset:expr;) => { [$($t,)*] };
    (@location [ $($t:expr,)* ]; $offset:expr; $comp:tt $(, $rest:tt)*) => {
        define_generic_vertex_types!(@location
            [
                $($t,)*
                VertexLocation {
                    offset: $offset,
                    format: <$comp as VertexAttribute>::FORMAT,
                    component: <$comp as VertexAttribute>::COMPONENT,
                },
            ];
            $offset + std::mem::size_of::<$comp>() as u32;
            $($rest),*
        )
    };
}

define_generic_vertex_types! {
    /// Vertex with 2 attributes.
    pub Vertex2(A1, A2);
    /// Vertex with 3 attributes.
    pub Vertex3(A1, A2, A3);
    /// Vertex with 4 attributes.
    pub Vertex4(A1, A2, A3, A4);
}

pub type Position2UV = Vertex2<Position2, UV>;
pub type Position2UVColor = Vertex3<Position2, UV, Color>;
pub type Position3UV = Vertex2<Position3, UV>;
pub type Position3NormalUV = Vertex3<Position3, Normal, UV>;
pub type Position3NormalTangentUV = Vertex4<Position3, Normal, Tangent, UV>;
