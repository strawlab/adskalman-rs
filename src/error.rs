/// An Kalman filter error
#[derive(Debug)]
pub enum Error {
    /// The covariance matrix is not positive semi-definite (or is not symmetric).
    CovarianceNotPositiveSemiDefinite,
}

#[cfg(feature = "std")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            Self::CovarianceNotPositiveSemiDefinite => {
                "The covariance matrix is not positive semi-definite (or is not symmetric)"
            }
        };
        f.write_str(s)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
