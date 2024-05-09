use na::base::dimension::U1;
use na::base::storage::Owned;
use na::{DimName, RealField};
use na::{Matrix, Vector};
use nalgebra as na;

/// State and covariance pair for a given estimate
#[derive(Debug, Clone)]
pub struct StateAndCovariance<R, SS, S1 = Owned<R, SS, U1>, S2 = Owned<R, SS, SS>>
where
    R: RealField,
    SS: DimName,
    S1: Clone,
    S2: Clone,
{
    state: Vector<R, SS, S1>,
    covariance: Matrix<R, SS, SS, S2>,
}

impl<R, SS, S1, S2> StateAndCovariance<R, SS, S1, S2>
where
    R: RealField,
    SS: DimName,
    S1: Clone,
    S2: Clone,
{
    /// Create a new `StateAndCovariance`.
    ///
    /// It is assumed that the covariance matrix is symmetric and positive
    /// semi-definite.
    pub fn new(state: Vector<R, SS, S1>, covariance: Matrix<R, SS, SS, S2>) -> Self {
        // In theory, checks could be run to ensure the covariance matrix is
        // both symmetric and positive semi-definite. The Cholesky decomposition
        // could be used to test if it is positive definite. However, matrices
        // which are positive semi-definite but not positive definite are also
        // valid covariance matrices. Thus, if the Cholesky decomposition fails,
        // the eigenvalues could be computed and used to test semi-definiteness
        // (e.g. https://scicomp.stackexchange.com/questions/12979).
        //
        // I have decided that the computational cost is not worth the marginal
        // benefits such testing would bring. If your covariance matrices might
        // not be symmetric and positive semi-definite, test them prior to this.
        Self { state, covariance }
    }
    /// Get a reference to the state vector.
    #[inline]
    pub fn state(&self) -> &Vector<R, SS, S1> {
        &self.state
    }
    /// Get a mut reference to the state vector.
    #[inline]
    pub fn state_mut(&mut self) -> &mut Vector<R, SS, S1> {
        &mut self.state
    }
    /// Get a reference to the covariance matrix.
    #[inline]
    pub fn covariance(&self) -> &Matrix<R, SS, SS, S2> {
        &self.covariance
    }
    /// Get a mutable reference to the covariance matrix.
    #[inline]
    pub fn covariance_mut(&mut self) -> &mut Matrix<R, SS, SS, S2> {
        &mut self.covariance
    }
    /// Get the state vector and covariance matrix.
    #[inline]
    pub fn inner(self) -> (Vector<R, SS, S1>, Matrix<R, SS, SS, S2>) {
        (self.state, self.covariance)
    }
}
