/// An error
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

/// The kinds of errors
#[derive(Debug)]
pub enum ErrorKind {
    /// The covariance matrix is not positive semi-definite (or is not symmetric).
    CovarianceNotPositiveSemiDefinite,
}

#[cfg(feature = "std")]
impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use ErrorKind::*;
        let s = match self {
            CovarianceNotPositiveSemiDefinite => {
                "The covariance matrix is not positive semi-definite (or is not symmetric)"
            }
        };
        f.write_str(s)
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Error {
        Error { kind }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
#[cfg(feature = "std")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Kalman Filter Error: {}", self.kind)
    }
}
