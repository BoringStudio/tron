use syn::{parse_macro_input, DeriveInput};

use self::as_shader_layout::{impl_as_shader_layout, LayoutType};

mod as_shader_layout;

#[proc_macro_derive(AsStd140)]
pub fn as_std140(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_as_shader_layout(input, LayoutType::Std140).into()
}

#[proc_macro_derive(AsStd430)]
pub fn as_std430(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_as_shader_layout(input, LayoutType::Std430).into()
}
