use nalgebra::{
    allocator::Allocator, base::storage::Owned, DefaultAllocator, Dim, Matrix, RealField, Vector,
};

/// State and covariance pair for a given estimate
#[derive(Debug, Clone)]
pub struct StateAndCovariance<R, SS>
where
    R: RealField,
    SS: Dim,
    DefaultAllocator: Allocator<SS> + Allocator<SS, SS>,
{
    state: Vector<R, SS, Owned<R, SS>>,
    covariance: Matrix<R, SS, SS, Owned<R, SS, SS>>,
}

impl<R, SS> StateAndCovariance<R, SS>
where
    R: RealField,
    SS: Dim,
    DefaultAllocator: Allocator<SS> + Allocator<SS, SS>,
{
    /// Create a new `StateAndCovariance`.
    ///
    /// It is assumed that the covariance matrix is symmetric and positive
    /// semi-definite.
    pub fn new(
        state: Vector<R, SS, Owned<R, SS>>,
        covariance: Matrix<R, SS, SS, Owned<R, SS, SS>>,
    ) -> Self {
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
    pub fn state(&self) -> &Vector<R, SS, Owned<R, SS>> {
        &self.state
    }
    /// Get a mut reference to the state vector.
    #[inline]
    pub fn state_mut(&mut self) -> &mut Vector<R, SS, Owned<R, SS>> {
        &mut self.state
    }
    /// Get a reference to the covariance matrix.
    #[inline]
    pub fn covariance(&self) -> &Matrix<R, SS, SS, Owned<R, SS, SS>> {
        &self.covariance
    }
    /// Get a mutable reference to the covariance matrix.
    #[inline]
    pub fn covariance_mut(&mut self) -> &mut Matrix<R, SS, SS, Owned<R, SS, SS>> {
        &mut self.covariance
    }
    /// Get the state vector and covariance matrix.
    #[inline]
    pub fn inner(
        self,
    ) -> (
        Vector<R, SS, Owned<R, SS>>,
        Matrix<R, SS, SS, Owned<R, SS, SS>>,
    ) {
        (self.state, self.covariance)
    }
}
