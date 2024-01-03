use once_cell::sync::OnceCell;

use anyhow::Result;
use shared::FastHashMap;

use self::virtual_fs::VirtualFs;

mod virtual_fs;

#[derive(Default)]
pub struct ShaderPreprocessor {
    fs: VirtualFs,
    global_defines: FastHashMap<String, Option<String>>,
    optimizations_enabled: bool,
}

impl ShaderPreprocessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file<T: AsRef<str>, V: Into<String>>(&mut self, path: T, contents: V) -> Result<()> {
        let path = path.as_ref();
        self.fs.add_file(path.as_ref(), contents.into())
    }

    pub fn define_global<T: Into<String>>(&mut self, name: T) {
        self.global_defines.insert(name.into(), None);
    }

    pub fn define_global_expr<T: Into<String>, V: Into<String>>(&mut self, name: T, value: V) {
        self.global_defines.insert(name.into(), Some(value.into()));
    }

    pub fn undefine_global<T: AsRef<str>>(&mut self, name: T) {
        self.global_defines.remove(name.as_ref());
    }

    pub fn set_optimizations_enabled(&mut self, enabled: bool) {
        self.optimizations_enabled = enabled;
    }

    pub fn begin(&self) -> ShaderPreprocessorScope<'_> {
        let mut res = ShaderPreprocessorScope {
            options: shaderc::CompileOptions::new().expect("failed to create `shaderc` options"),
        };

        res.options
            .set_include_callback(|include, _ty, source, depth| {
                if depth > 10 {
                    return Err("too many nested includes".to_string());
                }

                match self.fs.get_file(source.as_ref(), include.as_ref()) {
                    Ok(Some(file)) => Ok(shaderc::ResolvedInclude {
                        resolved_name: file.absolute_path,
                        content: file.contents.to_owned(),
                    }),
                    Ok(None) => Err(format!("file not found: {}", include)),
                    Err(err) => Err(format!("failed to read file: {}", err)),
                }
            });

        for (name, value) in &self.global_defines {
            match value {
                Some(value) => res.define_expr(name, value),
                None => res.define(name),
            }
        }
        res.set_optimizations_enabled(self.optimizations_enabled);
        res
    }
}

pub struct ShaderPreprocessorScope<'a> {
    options: shaderc::CompileOptions<'a>,
}

impl<'a> ShaderPreprocessorScope<'a> {
    pub fn define<T: AsRef<str>>(&mut self, name: T) {
        self.options.add_macro_definition(name.as_ref(), None)
    }

    pub fn define_expr<T: AsRef<str>, V: AsRef<str>>(&mut self, name: T, value: V) {
        self.options
            .add_macro_definition(name.as_ref(), Some(value.as_ref()));
    }

    pub fn set_optimizations_enabled(&mut self, enabled: bool) {
        self.options.set_optimization_level(if enabled {
            shaderc::OptimizationLevel::Performance
        } else {
            shaderc::OptimizationLevel::Zero
        });
    }

    pub fn compile_into_spirv<T: AsRef<str>, E: AsRef<str>>(
        &self,
        path: T,
        kind: gfx::ShaderType,
        entry: E,
    ) -> Result<Vec<u8>> {
        let path = self;

        // TODO
    }
}

fn shader_compiler() -> &'static shaderc::Compiler {
    static COMPILER: OnceCell<shaderc::Compiler> = OnceCell::new();
    COMPILER.get_or_init(|| shaderc::Compiler::new().expect("failed to create `shaderc` compiler"))
}
