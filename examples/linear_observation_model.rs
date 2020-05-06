use nalgebra as na;
use na::core::{VectorN, MatrixN, MatrixMN};
use na::core::dimension::{U2, U4};
use na::allocator::Allocator;
use na::core::dimension::DimMin;
use na::{DefaultAllocator, RealField};

use adskalman::ObservationModelLinear;

// observation model -------

pub(crate) struct PositionObservationModel<R: RealField>
    where DefaultAllocator: Allocator<R, U4, U4>,
          DefaultAllocator: Allocator<R, U2, U4>,
          DefaultAllocator: Allocator<R, U4, U2>,
          DefaultAllocator: Allocator<R, U2, U2>,
          DefaultAllocator: Allocator<R, U4>,
{
    pub(crate) observation_matrix: MatrixMN<R,U2,U4>,
    pub(crate) observation_matrix_transpose: MatrixMN<R,U4,U2>,
    pub(crate) observation_noise_covariance: MatrixN<R,U2>,
}

impl<R: RealField> PositionObservationModel<R> {
    #[allow(dead_code)]
    pub(crate) fn new(var: R) -> Self {
        let one = na::convert(1.0);
        let zero = na::convert(0.0);
        // Create observation model. We only observe the position.
        let observation_matrix = MatrixMN::<R,U2,U4>::new(one, zero, zero, zero,
                                    zero, one, zero, zero);
        let observation_noise_covariance = MatrixN::<R,U2>::new(var, zero,
                                                zero, var);
        Self {
            observation_matrix,
            observation_matrix_transpose: observation_matrix.transpose(),
            observation_noise_covariance,
        }
    }
}

impl<R: RealField> ObservationModelLinear<R, U4, U2> for PositionObservationModel<R>
    where DefaultAllocator: Allocator<R, U4, U4>,
          DefaultAllocator: Allocator<R, U2, U4>,
          DefaultAllocator: Allocator<R, U4, U2>,
          DefaultAllocator: Allocator<R, U2, U2>,
          DefaultAllocator: Allocator<R, U4>,
          DefaultAllocator: Allocator<R, U2>,
          DefaultAllocator: Allocator<(usize, usize), U2>,
          U2: DimMin<U2, Output = U2>,
{
    fn observation_matrix(&self) -> &MatrixMN<R,U2,U4> {
        &self.observation_matrix
    }
    fn observation_matrix_transpose(&self) -> &MatrixMN<R,U4,U2> {
        &self.observation_matrix_transpose
    }
    fn observation_noise_covariance(&self) -> &MatrixN<R,U2> {
        &self.observation_noise_covariance
    }
    fn evaluate(&self, state: &VectorN<R,U4>) -> VectorN<R,U2> {
        &self.observation_matrix * state
    }
}

#[allow(dead_code)]
fn main() {
    // TODO: can this .rs file be compiled just as a lib?
}
