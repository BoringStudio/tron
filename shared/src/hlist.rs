pub trait HList: Sized {
    fn prepend<H>(self, head: H) -> HCons<H, Self> {
        HCons { head, tail: self }
    }
}

#[derive(Debug, Default)]
pub struct HNil;

impl HList for HNil {}

pub struct HCons<H, T> {
    pub head: H,
    pub tail: T,
}

impl<H, T> HList for HCons<H, T> {}

impl<H: Default, T: Default> Default for HCons<H, T> {
    #[inline]
    fn default() -> Self {
        Self {
            head: H::default(),
            tail: T::default(),
        }
    }
}

impl<H: std::fmt::Debug, T: std::fmt::Debug> std::fmt::Debug for HCons<H, T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("HCons")
            .field(&self.head)
            .field(&self.tail)
            .finish()
    }
}

#[macro_export]
macro_rules! hlist_ty {
    ($($ty:ident),+) => { $crate::hlist_ty!(@inner [] [] $($ty)+) };

    (@inner [ $($prev:tt)* ] [ $($closing:tt)* ] $ty:ident) => {
        $($prev)* $crate::hlist::HCons<$ty, $crate::hlist::HNil>
        $($closing)*
    };
    (@inner [ $($prev:tt)* ] [ $($closing:tt)* ] $ty:ident $($rest:ident)+) => {
        $crate::hlist_ty!(@inner
            [$($prev)* $crate::hlist::HCons<$ty,]
            [$($closing)* >]
            $($rest)+
        )
    };
}

pub trait TupleToHList {
    type HList;

    fn into_hlist(self) -> Self::HList;
}

impl TupleToHList for () {
    type HList = HNil;

    #[inline]
    fn into_hlist(self) -> Self::HList {
        HNil
    }
}

macro_rules! impl_tuple_to_hlist {
    ($($idx:tt: $ty:ident),+$(,)?) => {
        impl<$($ty),*> TupleToHList for ($($ty),*,)
        {
            type HList = hlist_ty!($($ty),+);

            #[inline]
            fn into_hlist(self) -> Self::HList {
                impl_tuple_to_hlist!(@construct [ HNil ] [] $(self.$idx)+)
            }
        }
    };

    (@construct [ $($prev:tt)* ] []) => { $($prev)* };
    (@construct [ $($prev:tt)* ] [ $tuple:ident.$idx:tt $($rest:tt)* ]) => {
        impl_tuple_to_hlist!(@construct
            [HCons {
                head: $tuple.$idx,
                tail: $($prev)*,
            }]
            [ $($rest)* ]
        )
    };
    (@construct [ $($prev:tt)* ] [ $($reversed:tt)* ] $tuple:ident.$idx:tt $($rest:tt)* ) => {
        impl_tuple_to_hlist!(@construct
            [ $($prev)* ]
            [ $tuple.$idx $($reversed)*]
            $($rest)*
        )
    };
}

impl_tuple_to_hlist!(0: T0);
impl_tuple_to_hlist!(0: T0, 1: T1);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8);
impl_tuple_to_hlist!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9);

pub trait HListToTuple {
    type Tuple;

    fn into_tuple(self) -> Self::Tuple;
}

impl HListToTuple for HNil {
    type Tuple = ();

    #[inline]
    fn into_tuple(self) -> Self::Tuple {}
}

macro_rules! impl_hlist_to_tuple {
    ($($ty:ident),+$(,)?) => {
        impl<$($ty),*> HListToTuple for hlist_ty!($($ty),+)
        {
            type Tuple = ($($ty),*,);

            #[inline]
            fn into_tuple(self) -> Self::Tuple {
                impl_hlist_to_tuple!(@deconstruct [] [self] $($ty)+)
            }
        }
    };

    (@deconstruct [ $($prev:tt)* ] [ $($prefix:tt)* ]) => { ($($prev)*) };
    (@deconstruct [ $($prev:tt)* ] [ $($prefix:tt)* ] $ty:ident $($rest:ident)*) => {
        impl_hlist_to_tuple!(@deconstruct
            [$($prev)* $($prefix)*.head,]
            [$($prefix)*.tail]
            $($rest)*
        )
    };
}

impl_hlist_to_tuple!(T0);
impl_hlist_to_tuple!(T0, T1);
impl_hlist_to_tuple!(T0, T1, T2);
impl_hlist_to_tuple!(T0, T1, T2, T3);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6, T7);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
impl_hlist_to_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);

pub trait Selector<S, I> {
    fn get(&self) -> &S;
    fn get_mut(&mut self) -> &mut S;
}

impl<T, Tail> Selector<T, Here> for HCons<T, Tail> {
    #[inline]
    fn get(&self) -> &T {
        &self.head
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        &mut self.head
    }
}

impl<Head, Tail, T, I> Selector<T, There<I>> for HCons<Head, Tail>
where
    Tail: Selector<T, I>,
{
    #[inline]
    fn get(&self) -> &T {
        self.tail.get()
    }

    #[inline]
    fn get_mut(&mut self) -> &mut T {
        self.tail.get_mut()
    }
}

enum Here {}

struct There<T> {
    _marker: std::marker::PhantomData<T>,
}
