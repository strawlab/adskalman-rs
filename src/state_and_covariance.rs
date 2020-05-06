use nalgebra as na;
use na::core::{VectorN, MatrixN};
use na::{DefaultAllocator, DimName, RealField};
use na::allocator::Allocator;

/// State and covariance pair for a given estimate
#[derive(Debug, Clone)]
pub struct StateAndCovariance<R, SS>
    where
        R: RealField,
        SS: DimName,
        DefaultAllocator: Allocator<R, SS, SS>,
        DefaultAllocator: Allocator<R, SS>,
{
    state: VectorN<R, SS>,
    covariance: MatrixN<R, SS>,
}

impl<R, SS> StateAndCovariance<R, SS>
    where
        R: RealField,
        SS: DimName,
        DefaultAllocator: Allocator<R, SS, SS>,
        DefaultAllocator: Allocator<R, SS>,
{
    /// Create a new `StateAndCovariance`.
    ///
    /// It is assumed that the covariance matrix is symmetric and positive
    /// semi-definite.
    pub fn new(state: VectorN<R, SS>, covariance: MatrixN<R, SS>) -> Self {
        // In theory, checks could be run to ensure the covariance matrix is
        // both symmetric and positive semi-definite. The Cholesky decomposition
        // could be used to test if it is positive definite. However, matrices
        // which are positive semi-definite but not positive definite are also
        // valid covariance matrices. Thus, if the Cholesky decomposition fails,
        // the eigenvalues could be computed and used to test semi-definiteness
        // (e.g. https://scicomp.stackexchange.com/questions/12979).
        //
        // I have decided that the computational cost cost is not worth the
        // marginal benefits such testing would bring. If your covariance
        // matrices might not be symmetric and positive semi-definite, test them
        // prior to this.
        Self { state, covariance }
    }
    #[inline]
    pub fn state(&self) -> &VectorN<R, SS> {
        &self.state
    }
    #[inline]
    pub fn covariance(&self) -> &MatrixN<R, SS> {
        &self.covariance
    }
}
