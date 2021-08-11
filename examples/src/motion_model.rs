use nalgebra::{
    allocator::Allocator,
    convert,
    dimension::{U2, U4},
    DefaultAllocator, OMatrix, RealField,
};

use adskalman::TransitionModelLinearNoControl;

// motion model -------

#[allow(dead_code)]
pub struct ConstantVelocity2DModel<R>
where
    R: RealField,
    DefaultAllocator: Allocator<R, U4, U4>,
    DefaultAllocator: Allocator<R, U4>,
{
    pub transition_model: OMatrix<R, U4, U4>,
    pub transition_model_transpose: OMatrix<R, U4, U4>,
    pub transition_noise_covariance: OMatrix<R, U4, U4>,
}

impl<R> ConstantVelocity2DModel<R>
where
    R: RealField + Copy,
{
    #[allow(dead_code)]
    pub fn new(dt: R, noise_scale: R) -> Self {
        let one = convert(1.0);
        let zero = convert(0.0);
        // Create transition model. 2D position and 2D velocity.
        #[rustfmt::skip]
        let transition_model = OMatrix::<R,U4,U4>::new(one, zero,  dt, zero,
                            zero, one, zero,  dt,
                            zero, zero, one, zero,
                            zero, zero, zero, one);

        // This form is after N. Shimkin's lecture notes in
        // Estimation and Identification in Dynamical Systems
        // http://webee.technion.ac.il/people/shimkin/Estimation09/ch8_target.pdf
        // See also eq. 43 on pg. 13 of
        // http://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf

        let t33 = dt * dt * dt / convert(3.0);
        let t22 = dt * dt / convert(2.0);
        #[rustfmt::skip]
        let transition_noise_covariance = OMatrix::<R,U4,U4>::new(t33, zero, t22, zero,
                                        zero, t33, zero, t22,
                                        t22, zero, dt, zero,
                                        zero, t22, zero, dt)*noise_scale;
        Self {
            transition_model,
            transition_model_transpose: transition_model.transpose(),
            transition_noise_covariance,
        }
    }
}

impl<R> TransitionModelLinearNoControl<R, U4> for ConstantVelocity2DModel<R>
where
    R: RealField,
    DefaultAllocator: Allocator<R, U4, U4>,
    DefaultAllocator: Allocator<R, U2, U4>,
    DefaultAllocator: Allocator<R, U4, U2>,
    DefaultAllocator: Allocator<R, U2, U2>,
    DefaultAllocator: Allocator<R, U4>,
{
    fn F(&self) -> &OMatrix<R, U4, U4> {
        &self.transition_model
    }
    fn FT(&self) -> &OMatrix<R, U4, U4> {
        &self.transition_model_transpose
    }
    fn Q(&self) -> &OMatrix<R, U4, U4> {
        &self.transition_noise_covariance
    }
}
