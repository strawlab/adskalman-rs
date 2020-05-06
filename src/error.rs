#[derive(Debug)]
pub struct KalmanError {
    kind: ErrorKind,
}

#[derive(Debug)]
pub enum ErrorKind {
    /// The covariance matrix is not positive semi-definite (or is not symmetric).
    CovarianceNotPositiveSemiDefinite,
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use ErrorKind::*;
        let s = match self {
            CovarianceNotPositiveSemiDefinite => "The covariance matrix is not positive semi-definite (or is not symmetric)",
        };
        f.write_str(s)
    }
}

impl From<ErrorKind> for KalmanError {
    fn from(kind: ErrorKind) -> KalmanError {
        KalmanError { kind }
    }
}

impl std::error::Error for KalmanError {}
impl std::fmt::Display for KalmanError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Kalman Filter Error: {}", self.kind)
    }
}
