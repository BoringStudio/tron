use std::path::{Path, PathBuf};

use proc_macro::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Ident, LitStr, Token};

#[derive(Clone, Copy)]
enum ShaderKind {
    Unknown,

    Vertex,
    TesselationControl,
    TesselationEvaluation,
    Geometry,
    Fragment,
    Compute,
    // Mesh Pipeline
    Mesh,
    Task,
    // Ray-tracing Pipeline
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
}

struct ShaderCompilationConfig {
    incl_dirs: Vec<PathBuf>,
    defs: Vec<(String, Option<String>)>,
    entry: String,
    debug: bool,
    kind: ShaderKind,
    auto_bind: bool,
}

impl Default for ShaderCompilationConfig {
    fn default() -> Self {
        ShaderCompilationConfig {
            incl_dirs: Vec::new(),
            defs: Vec::new(),
            entry: "main".to_owned(),
            debug: true,
            kind: ShaderKind::Unknown,
            auto_bind: false,
        }
    }
}

struct CompilationFeedback {
    spv: Vec<u32>,
    dep_paths: Vec<String>,
}

struct InlineShaderSource(CompilationFeedback);
struct IncludedShaderSource(CompilationFeedback);

#[inline]
fn get_base_dir() -> PathBuf {
    let base_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("`spirv` can only be used in build time");
    PathBuf::from(base_dir)
}

#[inline]
fn parse_str(input: &mut ParseStream) -> syn::parse::Result<String> {
    input.parse::<LitStr>().map(|x| x.value())
}

#[inline]
fn parse_ident(input: &mut ParseStream) -> syn::parse::Result<String> {
    input.parse::<Ident>().map(|x| x.to_string())
}

fn parse_compile_cfg(input: &mut ParseStream) -> syn::parse::Result<ShaderCompilationConfig> {
    use syn::Error;

    let mut cfg = ShaderCompilationConfig::default();
    while !input.is_empty() {
        // Capture comma and collon; they are for readability.
        input.parse::<Token![,]>()?;
        let Ok(k) = input.parse::<Ident>() else {
            break;
        };

        match k.to_string().as_str() {
            "vert" => cfg.kind = ShaderKind::Vertex,
            "tesc" => cfg.kind = ShaderKind::TesselationControl,
            "tese" => cfg.kind = ShaderKind::TesselationEvaluation,
            "geom" => cfg.kind = ShaderKind::Geometry,
            "frag" => cfg.kind = ShaderKind::Fragment,
            "comp" => cfg.kind = ShaderKind::Compute,
            "mesh" => cfg.kind = ShaderKind::Mesh,
            "task" => cfg.kind = ShaderKind::Task,
            "rgen" => cfg.kind = ShaderKind::RayGeneration,
            "rint" => cfg.kind = ShaderKind::Intersection,
            "rahit" => cfg.kind = ShaderKind::AnyHit,
            "rchit" => cfg.kind = ShaderKind::ClosestHit,
            "rmiss" => cfg.kind = ShaderKind::Miss,
            "rcall" => cfg.kind = ShaderKind::Callable,

            "I" => cfg.incl_dirs.push(PathBuf::from(parse_str(input)?)),
            "D" => {
                let k = parse_ident(input)?;
                let v = if input.parse::<Token![=]>().is_ok() {
                    Some(parse_str(input)?)
                } else {
                    None
                };
                cfg.defs.push((k, v));
            }

            "entry" => {
                if input.parse::<Token![=]>().is_ok() {
                    cfg.entry = parse_str(input)?.to_owned();
                }
            }

            "no_debug" => cfg.debug = false,

            "auto_bind" => cfg.auto_bind = true,

            _ => return Err(Error::new(k.span(), "unsupported compilation parameter")),
        }
    }
    Ok(cfg)
}

fn compile(src: &str, cfg: &ShaderCompilationConfig) -> Result<CompilationFeedback, String> {
    use naga::{
        back::spv::WriterFlags,
        valid::{Capabilities, ValidationFlags, Validator},
    };

    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;

    let mut opts = naga::back::spv::Options::default();
    opts.lang_version = (1, 0);

    if cfg.debug {
        opts.flags.insert(WriterFlags::DEBUG);
    } else {
        opts.flags.remove(WriterFlags::DEBUG);
    }

    opts.flags.remove(WriterFlags::ADJUST_COORDINATE_SPACE);

    // Attempt to validate WGSL, error if invalid
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| format!("{:?}", e))?;
    let spv =
        naga::back::spv::write_vec(&module, &info, &opts, None).map_err(|e| format!("{:?}", e))?;

    Ok(CompilationFeedback {
        spv,
        dep_paths: Vec::new(),
    })
}

fn build_spirv_binary(path: &Path) -> Option<Vec<u32>> {
    use std::fs::File;
    use std::io::Read;

    let mut buf = Vec::new();
    if let Ok(mut f) = File::open(&path) {
        if buf.len() & 3 != 0 {
            // Misaligned input.
            return None;
        }
        f.read_to_end(&mut buf).ok()?;
    }

    let out = buf
        .chunks_exact(4)
        .map(|x| x.try_into().unwrap())
        .map(match buf[0] {
            0x03 => u32::from_le_bytes,
            0x07 => u32::from_be_bytes,
            _ => return None,
        })
        .collect::<Vec<u32>>();

    Some(out)
}

impl Parse for IncludedShaderSource {
    fn parse(mut input: ParseStream) -> syn::parse::Result<Self> {
        use std::ffi::OsStr;
        let path_lit = input.parse::<LitStr>()?;
        let path = Path::new(&get_base_dir()).join(&path_lit.value());

        if !path.is_file() {
            return Err(syn::parse::Error::new(
                path_lit.span(),
                format!("{path} is not a valid source file", path = path_lit.value()),
            ));
        }

        let feedback = if path.extension() == Some(OsStr::new("spv")) {
            let spv = build_spirv_binary(&path)
                .ok_or_else(|| syn::Error::new(path_lit.span(), "invalid spirv"))?;
            CompilationFeedback {
                spv,
                dep_paths: vec![],
            }
        } else {
            let src =
                std::fs::read_to_string(&path).map_err(|e| syn::Error::new(path_lit.span(), e))?;
            let cfg = parse_compile_cfg(&mut input)?;
            compile(&src, &cfg).map_err(|e| syn::parse::Error::new(input.span(), e))?
        };

        Ok(IncludedShaderSource(feedback))
    }
}

impl Parse for InlineShaderSource {
    fn parse(mut input: ParseStream) -> syn::parse::Result<Self> {
        let src = parse_str(&mut input)?;
        let cfg = parse_compile_cfg(&mut input)?;
        let feedback = compile(&src, &cfg).map_err(|e| syn::parse::Error::new(input.span(), e))?;
        Ok(InlineShaderSource(feedback))
    }
}

fn gen_token_stream(feedback: CompilationFeedback) -> TokenStream {
    let CompilationFeedback { spv, dep_paths } = feedback;
    (quote::quote! {
        {
            { #(let _ = include_bytes!(#dep_paths);)* }
            &[#(#spv),*]
        }
    })
    .into()
}

/// Compile inline shader source and embed the SPIR-V binary word sequence.
/// Returns a `&'static [u32]`.
#[proc_macro]
pub fn inline(tokens: TokenStream) -> TokenStream {
    let InlineShaderSource(feedback) = parse_macro_input!(tokens as InlineShaderSource);
    gen_token_stream(feedback)
}

/// Compile external shader source and embed the SPIR-V binary word sequence.
/// Returns a `&'static [u32]`.
#[proc_macro]
pub fn include(tokens: TokenStream) -> TokenStream {
    let IncludedShaderSource(feedback) = parse_macro_input!(tokens as IncludedShaderSource);
    gen_token_stream(feedback)
}
