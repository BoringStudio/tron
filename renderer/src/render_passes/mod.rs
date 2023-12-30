use anyhow::Result;

pub use self::main_pass::{MainPass, MainPassInput};

mod main_pass;

pub trait EncoderExt {
    fn with_render_pass<'a, 'b, P>(
        &'a mut self,
        pass: &'b mut P,
        input: &P::Input,
        device: &gfx::Device,
    ) -> Result<gfx::RenderPassEncoder<'a, 'b>>
    where
        P: Pass;
}

impl EncoderExt for gfx::Encoder {
    fn with_render_pass<'a, 'b, P>(
        &'a mut self,
        pass: &'b mut P,
        input: &P::Input,
        device: &gfx::Device,
    ) -> Result<gfx::RenderPassEncoder<'a, 'b>>
    where
        P: Pass,
    {
        pass.begin_render_pass(input, device, self)
    }
}

pub trait Pass {
    type Input;

    fn begin_render_pass<'a, 'b>(
        &'b mut self,
        input: &Self::Input,
        device: &gfx::Device,
        encoder: &'a mut gfx::Encoder,
    ) -> Result<gfx::RenderPassEncoder<'a, 'b>>;
}
