use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Type};

pub fn impl_as_shader_layout(input: DeriveInput, layout_type: LayoutType) -> TokenStream {
    let trait_name = format_ident!("{}", layout_type.name());
    let trait_path = quote! { ::gfx::#trait_name };

    let as_trait_name = format_ident!("As{}", layout_type.name());
    let as_trait_path = quote! { ::gfx::#as_trait_name };
    let as_trait_method = layout_type.as_trait_method();
    let write_as_trait_method = layout_type.write_as_trait_method();

    let visibility = input.vis;
    let input_name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let generated_name = format_ident!("{}{}", layout_type.name(), input_name);

    let fields: Vec<_> = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => fields.named.iter().collect(),
            Fields::Unnamed(_) => panic!("Tuple structs are not supported"),
            Fields::Unit => panic!("Unit structs are not supported"),
        },
        _ => panic!("Only structs are supported"),
    };

    let layout_version_of_ty = |ty: &Type| {
        quote! { <#ty as #as_trait_path>::Output }
    };
    let layout_align_mask_of_ty = |ty: &Type| {
        quote! { <<#ty as #as_trait_path>::Output as #trait_path>::ALIGN_MASK }
    };

    let min_align_mask = layout_type.min_align_mask();
    let field_alginment = fields
        .iter()
        .map(|field| layout_align_mask_of_ty(&field.ty));
    let struct_alignment = quote! {
        #min_align_mask #( | #field_alginment )*
    };

    // Generate names for each padding calculation function.
    let pad_fns: Vec<_> = (0..fields.len())
        .map(|index| format_ident!("_{}__{}Pad{}", input_name, trait_name, index))
        .collect();

    // Computes the offset immediately AFTER the field with the given index.
    //
    // This function depends on the generated padding calculation functions to
    // do correct alignment. Be careful not to cause recursion!
    let offset_after_field = |target: usize| {
        let mut output = vec![quote!(0usize)];

        for index in 0..=target {
            let layout_ty = layout_version_of_ty(&fields[index].ty);
            output.push(quote! {
                + ::core::mem::size_of::<#layout_ty>()
            });

            // For every field except our target field, also add the generated
            // padding. Padding occurs after each field, so it isn't included in
            // this value.
            if index < target {
                let pad_fn = &pad_fns[index];
                output.push(quote! {
                    + #pad_fn()
                });
            }
        }

        output.into_iter().collect::<TokenStream>()
    };

    let pad_fn_impls: TokenStream = pad_fns
        .iter()
        .enumerate()
        .map(|(index, pad_fn)| {
            let starting_offset = offset_after_field(index);

            let next_field_or_self_align_mask = fields
                .get(index + 1)
                .map(|next_field| layout_align_mask_of_ty(&next_field.ty))
                .unwrap_or(quote!(#struct_alignment));

            quote! {
                #[allow(non_snake_case)]
                const fn #pad_fn() -> usize {
                    let align_mask = #next_field_or_self_align_mask;
                    let offset = #starting_offset;
                    ::gfx::align_offset(align_mask, offset) as usize
                }
            }
        })
        .collect();

    let generated_struct_fields: TokenStream = fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let field_name = field.ident.as_ref().unwrap();
            let field_ty = layout_version_of_ty(&field.ty);
            let pad_field_name = format_ident!("_pad{}", index);
            let pad_fn = &pad_fns[index];

            quote! {
                #field_name: #field_ty,
                #pad_field_name: [u8; #pad_fn()],
            }
        })
        .collect();

    let doc: TokenStream = format!(
        "#[doc = \"A structure to use [`{}`] in shader with {} compatible layout\"]",
        input_name,
        layout_type.name()
    )
    .parse()
    .unwrap();

    let struct_definition = quote! {
        #doc
        #[derive(Debug, Clone, Copy)]
        #[repr(C)]
        #visibility struct #generated_name #ty_generics #where_clause {
            #generated_struct_fields
        }
    };

    let as_trait_fields: TokenStream = fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            quote! {
                #field_name: self.#field_name.#as_trait_method(),
            }
        })
        .collect();

    let write_as_trait_fields: TokenStream = fields
        .iter()
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            quote! {
                dst.#field_name = self.#field_name.#as_trait_method();
            }
        })
        .collect();

    quote! {
        #struct_definition
        #pad_fn_impls

        unsafe impl #impl_generics ::gfx::inner_proc_stuff::bytemuck::Zeroable for #generated_name #ty_generics #where_clause {}
        unsafe impl #impl_generics ::gfx::inner_proc_stuff::bytemuck::Pod for #generated_name #ty_generics #where_clause {}

        impl #impl_generics #generated_name #ty_generics #where_clause {
            pub fn as_bytes(&self) -> &[u8] {
                <#generated_name #ty_generics as #trait_path>::as_bytes(self)
            }
        }

        unsafe impl #impl_generics #trait_path for #generated_name #ty_generics #where_clause {
            const ALIGN_MASK: usize = #struct_alignment;

            type ArrayPadding = [u8; 0];
        }

        impl #impl_generics #as_trait_path for #input_name #ty_generics #where_clause {
            type Output = #generated_name;

            fn #as_trait_method(&self) -> Self::Output {
                Self::Output {
                    #as_trait_fields
                    ..::gfx::inner_proc_stuff::bytemuck::Zeroable::zeroed()
                }
            }

            fn #write_as_trait_method(&self, dst: &mut Self::Output) {
                #write_as_trait_fields
            }
        }
    }
}

pub enum LayoutType {
    Std140,
    Std430,
}

impl LayoutType {
    fn name(&self) -> &'static str {
        match self {
            LayoutType::Std140 => "Std140",
            LayoutType::Std430 => "Std430",
        }
    }

    fn as_trait_method(&self) -> TokenStream {
        match self {
            LayoutType::Std140 => quote! { as_std140 },
            LayoutType::Std430 => quote! { as_std430 },
        }
    }

    fn write_as_trait_method(&self) -> TokenStream {
        match self {
            LayoutType::Std140 => quote! { write_as_std140 },
            LayoutType::Std430 => quote! { write_as_std430 },
        }
    }

    fn min_align_mask(&self) -> usize {
        match self {
            LayoutType::Std140 => 0b1111,
            LayoutType::Std430 => 0,
        }
    }
}
