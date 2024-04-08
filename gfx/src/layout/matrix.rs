use glam::{Affine2, Affine3A, Mat2, Mat3, Mat4, Vec2, Vec3, Vec4};

use super::{AsStd140, AsStd430};

impl AsStd140 for Mat2 {
    type Output = <[Vec2; 2] as AsStd140>::Output;

    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
    }
}

impl AsStd430 for Mat2 {
    type Output = <[Vec2; 2] as AsStd430>::Output;

    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
    }
}

impl AsStd140 for Mat3 {
    type Output = <[Vec3; 3] as AsStd140>::Output;

    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
    }
}

impl AsStd430 for Mat3 {
    type Output = <[Vec3; 3] as AsStd430>::Output;

    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
        dst[2].value = self.z_axis;
    }
}

impl AsStd140 for Mat4 {
    type Output = <[Vec4; 4] as AsStd140>::Output;

    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
        dst[2].value = self.z_axis;
        dst[3].value = self.w_axis;
    }
}

impl AsStd430 for Mat4 {
    type Output = <[Vec4; 4] as AsStd430>::Output;

    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
        dst[2].value = self.z_axis;
        dst[3].value = self.w_axis;
    }
}

impl AsStd140 for Affine2 {
    type Output = <[Vec2; 3] as AsStd140>::Output;

    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
        dst[2].value = self.translation;
    }
}

impl AsStd430 for Affine2 {
    type Output = <[Vec2; 3] as AsStd430>::Output;

    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst[0].value = self.x_axis;
        dst[1].value = self.y_axis;
        dst[2].value = self.translation;
    }
}

impl AsStd140 for Affine3A {
    type Output = <[Vec3; 4] as AsStd140>::Output;

    fn write_as_std140(&self, dst: &mut Self::Output) {
        dst[0].value = Vec3::from(self.x_axis);
        dst[1].value = Vec3::from(self.y_axis);
        dst[2].value = Vec3::from(self.z_axis);
        dst[3].value = Vec3::from(self.translation);
    }
}

impl AsStd430 for Affine3A {
    type Output = <[Vec3; 4] as AsStd430>::Output;

    fn write_as_std430(&self, dst: &mut Self::Output) {
        dst[0].value = Vec3::from(self.x_axis);
        dst[1].value = Vec3::from(self.y_axis);
        dst[2].value = Vec3::from(self.z_axis);
        dst[3].value = Vec3::from(self.translation);
    }
}
