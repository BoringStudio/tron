use std::borrow::Cow;

use once_cell::sync::OnceCell;

use anyhow::Result;
use shared::FastHashMap;

use self::virtual_fs::{VirtualFs, VirtualPath};

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

    pub fn add_file(
        &mut self,
        path: impl AsRef<str>,
        contents: impl Into<Cow<'static, str>>,
    ) -> Result<()> {
        self.fs.add_file(path.as_ref(), contents)
    }

    pub fn define_global(&mut self, name: impl Into<String>) {
        self.global_defines.insert(name.into(), None);
    }

    pub fn define_global_expr(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.global_defines.insert(name.into(), Some(value.into()));
    }

    pub fn undefine_global(&mut self, name: impl AsRef<str>) {
        self.global_defines.remove(name.as_ref());
    }

    pub fn set_optimizations_enabled(&mut self, enabled: bool) {
        self.optimizations_enabled = enabled;
    }

    pub fn begin(&self) -> ShaderPreprocessorScope<'_> {
        let mut res = ShaderPreprocessorScope {
            inner: self,
            options: shaderc::CompileOptions::new().expect("failed to create `shaderc` options"),
        };

        res.options
            .set_include_callback(|include, _ty, source, depth| {
                if depth > 10 {
                    return Err("too many nested includes".to_string());
                }

                match self.fs.get_file(source, include) {
                    Ok(Some(file)) => Ok(shaderc::ResolvedInclude {
                        resolved_name: file.absolute_path,
                        content: file.contents.to_owned(),
                    }),
                    Ok(None) => Err("file not found".to_owned()),
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
    inner: &'a ShaderPreprocessor,
    options: shaderc::CompileOptions<'a>,
}

impl<'a> ShaderPreprocessorScope<'a> {
    pub fn define<T: AsRef<str>>(&mut self, name: T) {
        self.options.add_macro_definition(name.as_ref(), None)
    }

    pub fn define_expr(&mut self, name: impl AsRef<str>, value: impl AsRef<str>) {
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

    pub fn make_vertex_shader(
        &self,
        device: &gfx::Device,
        path: impl AsRef<str>,
        entry: impl AsRef<str>,
    ) -> Result<gfx::VertexShader> {
        let module = self.make_shader_module(
            device,
            path.as_ref(),
            entry.as_ref(),
            gfx::ShaderType::Vertex,
        )?;
        Ok(gfx::VertexShader::new(module, entry.as_ref().to_owned()))
    }

    pub fn make_fragment_shader(
        &self,
        device: &gfx::Device,
        path: impl AsRef<str>,
        entry: impl AsRef<str>,
    ) -> Result<gfx::FragmentShader> {
        let module = self.make_shader_module(
            device,
            path.as_ref(),
            entry.as_ref(),
            gfx::ShaderType::Fragment,
        )?;
        Ok(gfx::FragmentShader::new(module, entry.as_ref().to_owned()))
    }

    pub fn make_compute_shader(
        &self,
        device: &gfx::Device,
        path: impl AsRef<str>,
        entry: impl AsRef<str>,
    ) -> Result<gfx::ComputeShader> {
        let module = self.make_shader_module(
            device,
            path.as_ref(),
            entry.as_ref(),
            gfx::ShaderType::Compute,
        )?;
        Ok(gfx::ComputeShader::new(module, entry.as_ref().to_owned()))
    }

    pub fn make_shader_module(
        &self,
        device: &gfx::Device,
        path: impl AsRef<str>,
        entry: impl AsRef<str>,
        shader_type: gfx::ShaderType,
    ) -> Result<gfx::ShaderModule> {
        self.make_shader_module_impl(device, path.as_ref(), entry.as_ref(), shader_type)
    }

    fn make_shader_module_impl(
        &self,
        device: &gfx::Device,
        path: &str,
        entry: &str,
        shader_type: gfx::ShaderType,
    ) -> Result<gfx::ShaderModule> {
        let info = self.compile_shader(path.as_ref(), entry.as_ref(), shader_type)?;
        device.create_shader_module(info)
    }

    fn compile_shader(
        &self,
        path: &str,
        entry: &str,
        shader_type: gfx::ShaderType,
    ) -> Result<gfx::ShaderModuleInfo> {
        let fs = &self.inner.fs;
        let Some(file) = fs.get_file(VirtualPath::root(), VirtualPath::new(path))? else {
            anyhow::bail!("file not found: {path}");
        };

        let shader_type = match shader_type {
            gfx::ShaderType::Vertex => shaderc::ShaderKind::Vertex,
            gfx::ShaderType::Fragment => shaderc::ShaderKind::Fragment,
            gfx::ShaderType::Compute => shaderc::ShaderKind::Compute,
        };

        let data = shader_compiler().compile_into_spirv(
            file.contents,
            shader_type,
            &file.absolute_path,
            entry,
            Some(&self.options),
        )?;
        if data.get_num_warnings() > 0 {
            tracing::warn!(
                ?shader_type,
                path = file.absolute_path,
                "{}",
                data.get_warning_messages()
            );
        }

        Ok(gfx::ShaderModuleInfo {
            data: Box::from(data.as_binary()),
        })
    }
}

fn shader_compiler() -> &'static shaderc::Compiler {
    static COMPILER: OnceCell<shaderc::Compiler> = OnceCell::new();
    COMPILER.get_or_init(|| shaderc::Compiler::new().expect("failed to create `shaderc` compiler"))
}
