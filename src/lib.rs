#![cfg_attr(not(feature = "std"), no_std)]
//! Kalman filter and Rauch-Tung-Striebel smoothing implementation using nalgebra, `no_std`
//!
//! See [our examples](https://github.com/strawlab/adskalman-rs/tree/master/examples).

// Ideas for improvement:
//  - See http://mocha-java.uccs.edu/ECE5550/, especially
//    "5.1: Maintaining symmetry of covariance matrices".
//  - See http://www.anuncommonlab.com/articles/how-kalman-filters-work/part2.html
//  - See https://stats.stackexchange.com/questions/67262/non-overlapping-state-and-measurement-covariances-in-kalman-filter/292690
//  - https://en.wikipedia.org/wiki/Kalman_filter#Square_root_form

#[cfg(feature="std")]
use log::trace;
#[cfg(debug_assertions)]
use approx::assert_relative_eq;
use nalgebra as na;
use na::{VectorN, MatrixN, MatrixMN};
use nalgebra::base::dimension::DimMin;

use na::{DefaultAllocator, DimName, RealField};
use na::allocator::Allocator;

use num_traits::identities::One;

// Without std, create a dummy trace!() macro.
#[cfg(not(feature="std"))]
macro_rules! trace {
    ($e:expr) => {{}};
    ($e:expr, $($es:expr),+) => {{}};
}

/// perform a runtime check that matrix is symmetric
///
/// only compiled in debug mode
macro_rules! debug_assert_symmetric {
    ($mat:expr) => {
        #[cfg(debug_assertions)]
        {
            assert_relative_eq!( $mat, &$mat.transpose(),
                max_relative = na::convert(1e-5));
        }
    }
}

/// convert an nalgebra array to a String
#[cfg(feature="std")]
macro_rules! pretty_print {
    ($arr:expr) => {{
        let indent = 4;
        let prefix = String::from_utf8(vec![b' '; indent]).unwrap();
        let mut result_els = vec!["".to_string()];
        for i in 0..$arr.nrows() {
            let mut row_els = vec![];
            for j in 0..$arr.ncols() {
                row_els.push(format!("{:12.3}", $arr[(i,j)]));
            }
            let row_str = row_els.into_iter().collect::<Vec<_>>().join(" ");
            let row_str = format!("{}{}", prefix, row_str);
            result_els.push( row_str );
        }
        result_els.into_iter().collect::<Vec<_>>().join("\n")
    }}
}


mod error;
pub use error::{KalmanError, ErrorKind};

mod state_and_covariance;
pub use state_and_covariance::StateAndCovariance;

/// A linear model of process dynamics with no control inputs
pub trait TransitionModelLinearNoControl<R, SS>
    where
        R: RealField,
        SS: DimName,
        DefaultAllocator: Allocator<R, SS, SS>,
        DefaultAllocator: Allocator<R, SS>,
{
    fn transition_model(&self) -> &MatrixN<R, SS>;
    fn transition_model_transpose(&self) -> &MatrixN<R, SS>;
    fn transition_noise_covariance(&self) -> &MatrixN<R, SS>;
    fn predict(&self, previous_estimate: &StateAndCovariance<R, SS>) -> StateAndCovariance<R, SS> {
        let state = self.transition_model() * previous_estimate.state();
        let covariance = ((self.transition_model() * previous_estimate.covariance()) *
                              self.transition_model_transpose()) +
            self.transition_noise_covariance();
        StateAndCovariance::new(state, covariance)
    }
}

/// A linear observation model
///
/// Note, to use a non-linear observation model, the non-linear model must
/// be linearized (using the prior state estimate) and use this linearization
/// as the basis for a `ObservationModelLinear` implementation.
pub trait ObservationModelLinear<R, SS, OS>
    where
        R: RealField,
        SS: DimName,
        OS: DimName + DimMin<OS, Output = OS>,
        DefaultAllocator: Allocator<R, SS, SS>,
        DefaultAllocator: Allocator<R, SS>,
        DefaultAllocator: Allocator<R, OS, SS>,
        DefaultAllocator: Allocator<R, SS, OS>,
        DefaultAllocator: Allocator<R, OS, OS>,
        DefaultAllocator: Allocator<R, OS>,
        DefaultAllocator: Allocator<(usize, usize), OS>,
{
    fn evaluate(&self, state: &VectorN<R,SS>) -> VectorN<R,OS>;

    fn observation_matrix(&self) -> &MatrixMN<R,OS,SS>;
    fn observation_matrix_transpose(&self) -> &MatrixMN<R,SS,OS>;

    // TODO: ensure this is positive definite?
    fn observation_noise_covariance(&self) -> &MatrixN<R,OS>;

    fn update(&self,
        prior: &StateAndCovariance<R,SS>,
        observation: &VectorN<R,OS>,
        covariance_method: CoverianceUpdateMethod,
    )
        -> StateAndCovariance<R,SS>
    {
        // Use conventional (e.g. wikipedia) names for these variables
        let h = self.observation_matrix();
        trace!("h {}", pretty_print!(h));

        let p = prior.covariance();
        trace!("p {}", pretty_print!(p));
        debug_assert_symmetric!(p);

        let ht = self.observation_matrix_transpose();
        trace!("ht {}", pretty_print!(ht));

        let r = self.observation_noise_covariance();
        trace!("r {}", pretty_print!(r));

        // Calculate innovation covariance
        //
        // Math note: if (h*p*ht) and r are positive definite, s is also
        // positive definite. If p is positive definite, then (h*p*ht) is at
        // least positive semi-definite. If h is full rank, it is positive
        // definite.
        let s = (h*p*ht)+r;
        trace!("s {}", pretty_print!(s));

        // Calculate kalman gain by inverting.
        let s_chol = match na::linalg::Cholesky::new(s) {
            Some(v) => v,
            None => {
                // Maybe state covariance is not symmetric or
                // for from positive definite? Also, observation
                // noise should be positive definite.
                panic!("Innovation covariance not positive definite.");
            }
        };
        let s_inv: MatrixMN<R,OS,OS> = s_chol.inverse();
        trace!("s_inv {}", pretty_print!(s_inv));

        let k_gain: MatrixMN<R,SS,OS> = p * ht * s_inv;
        // let k_gain: MatrixMN<R,SS,OS> = solve!( (p*ht), s );
        trace!("k_gain {}", pretty_print!(k_gain));

        let predicted: VectorN<R,OS> = self.evaluate(prior.state());
        trace!("predicted {}", pretty_print!(predicted));
        trace!("observation {}", pretty_print!(observation));
        let innovation: VectorN<R,OS> = observation - predicted;
        trace!("innovation {}", pretty_print!(innovation));
        let state: VectorN<R,SS> = prior.state() + &k_gain * innovation;
        trace!("state {}", pretty_print!(state));

        trace!("self.observation_matrix() {}", pretty_print!(self.observation_matrix()));
        let kh: MatrixN<R,SS> = &k_gain * self.observation_matrix();
        trace!("kh {}", pretty_print!(kh));
        let one_minus_kh = MatrixN::<R,SS>::one() - kh;
        trace!("one_minus_kh {}", pretty_print!(one_minus_kh));

        let covariance: MatrixN<R,SS> = match covariance_method {
            CoverianceUpdateMethod::JosephForm => {
                // Joseph form of covariance update keeps covariance matrix symmetric.

                let left = &one_minus_kh * prior.covariance() * &one_minus_kh.transpose();
                let right = &k_gain * r * &k_gain.transpose();
                left + right
            }
            CoverianceUpdateMethod::OptimalKalman => {
                one_minus_kh * prior.covariance()
            }
            CoverianceUpdateMethod::OptimalKalmanForcedSymmetric => {
                let covariance1 = one_minus_kh * prior.covariance();
                trace!("covariance1 {}", pretty_print!(covariance1));

                // Hack to force covariance to be symmetric.
                // See https://math.stackexchange.com/q/2335831
                let half: R = na::convert(0.5);
                (&covariance1 + &covariance1.transpose()) * half
            }
        };
        trace!("covariance {}", pretty_print!(covariance));

        debug_assert_symmetric!(covariance);

        StateAndCovariance::new(state, covariance)
    }
}

/// Specifies the approach used for updating the covariance matrix
#[derive(Debug,PartialEq,Clone,Copy)]
pub enum CoverianceUpdateMethod {
    /// Assumes optimal Kalman gain.
    ///
    /// Due to numerical errors, covariance matrix may not remain symmetric.
    OptimalKalman,
    /// Assumes optimal Kalman gain and then forces symmetric covariance matrix.
    ///
    /// With original covariance matrix P, returns covariance as (P + P.T)/2
    /// to enforce that the covariance matrix remains symmetric.
    OptimalKalmanForcedSymmetric,
    /// Joseph form of covariance update keeps covariance matrix symmetric.
    JosephForm,
}

/// A Kalman filter with no control inputs, a linear process model and linear observation model
pub struct KalmanFilterNoControl<'a, R, SS, OS>
    where
        R: RealField,
        SS: DimName,
        OS: DimName,
{
    transition_model: &'a dyn TransitionModelLinearNoControl<R, SS>,
    observation_matrix: &'a dyn ObservationModelLinear<R, SS, OS>,
}

impl<'a, R, SS, OS> KalmanFilterNoControl<'a, R, SS, OS>
    where
        R: RealField,
        SS: DimName,
        OS: DimName + DimMin<OS, Output = OS>,
        DefaultAllocator: Allocator<R, SS, SS>,
        DefaultAllocator: Allocator<R, SS>,
        DefaultAllocator: Allocator<R, OS, SS>,
        DefaultAllocator: Allocator<R, SS, OS>,
        DefaultAllocator: Allocator<R, OS, OS>,
        DefaultAllocator: Allocator<R, OS>,
        DefaultAllocator: Allocator<(usize, usize), OS>,
{
    pub fn new(
        transition_model: &'a dyn TransitionModelLinearNoControl<R, SS>,
        observation_matrix: &'a dyn ObservationModelLinear<R, SS, OS>,
    ) -> Self {
        Self {
            transition_model,
            observation_matrix,
        }
    }

    /// Perform Kalman prediction and update steps with default values
    ///
    /// If any component of the observation is NaN (not a number), the
    /// observation will not be used but rather the prior will be returned as
    /// the posterior without performing the update step.
    ///
    /// This calls the prediction step of the transition model and then, if
    /// there is a (non-`nan`) observation, calls the update step of the observation
    /// model using the `CoverianceUpdateMethod::OptimalKalmanForcedSymmetric`
    /// covariance update method.
    ///
    /// This is a convenience function calling [step_with_options](struct.KalmanFilterNoControl.html#method.step_with_options)
    pub fn step(&self, previous_estimate: &StateAndCovariance<R,SS>, observation: &VectorN<R,OS>)
        -> StateAndCovariance<R,SS>
    {
        self.step_with_options(previous_estimate, observation, CoverianceUpdateMethod::OptimalKalmanForcedSymmetric)
    }

    /// Perform Kalman prediction and update steps with default values
    ///
    /// If any component of the observation is NaN (not a number), the
    /// observation will not be used but rather the prior will be returned as
    /// the posterior without performing the update step.
    ///
    /// This calls the prediction step of the transition model and then, if
    /// there is a (non-`nan`) observation, calls the update step of the
    /// observation model using the specified covariance update method.
    pub fn step_with_options(&self,
        previous_estimate: &StateAndCovariance<R,SS>,
        observation: &VectorN<R,OS>,
        covariance_update_method: CoverianceUpdateMethod,
    )
        -> StateAndCovariance<R,SS>
    {
        let prior = self.transition_model.predict(previous_estimate);
        if observation.iter().any(|x| is_nan(*x)) {
            prior
        } else {
            self.observation_matrix.update(
                &prior,
                observation,
                covariance_update_method,
            )
        }
    }

    /// Kalman filter (operates on in-place data without allocating)
    ///
    /// Operates on entire time series in one shot and returns a vector of state
    /// estimates. To be mathematically correct, the interval between
    /// observations must be the `dt` specified in the motion model.
    ///
    /// If any observation has a NaN component, it is treated as missing.
    pub fn filter_inplace(&self, initial_estimate: &StateAndCovariance<R,SS>, observations: &[VectorN<R,OS>], state_estimates: &mut [StateAndCovariance<R,SS>]) {
        let mut previous_estimate = initial_estimate.clone();
        assert!(state_estimates.len() >= observations.len());

        for (this_observation, state_estimate) in observations.iter().zip(state_estimates.iter_mut()) {
            let this_estimate = self.step(&previous_estimate, this_observation);
            *state_estimate = this_estimate.clone();
            previous_estimate = this_estimate;
        }
    }

    /// Kalman filter
    ///
    /// This is a convenience function that calls [`filter_inplace`](struct.KalmanFilterNoControl.html#method.filter_inplace).
    #[cfg(feature="std")]
    pub fn filter(&self, initial_estimate: &StateAndCovariance<R,SS>, observations: &[VectorN<R,OS>]) -> Vec<StateAndCovariance<R,SS>> {

        let mut state_estimates = Vec::with_capacity(observations.len());
        let empty = StateAndCovariance::new(na::zero(), na::MatrixN::<R,SS>::identity());
        for _ in 0..observations.len() {
            state_estimates.push(empty.clone());
        }
        self.filter_inplace(initial_estimate, observations, &mut state_estimates);
        state_estimates
    }

    /// Rauch-Tung-Striebel (RTS) smoother
    ///
    /// Operates on entire time series in one shot and returns a vector of state
    /// estimates. To be mathematically correct, the interval between
    /// observations must be the `dt` specified in the motion model.
    ///
    /// If any observation has a NaN component, it is treated as missing.
    #[cfg(feature="std")]
    pub fn smooth(&self, initial_estimate: &StateAndCovariance<R,SS>, observations: &[VectorN<R,OS>]) -> Vec<StateAndCovariance<R,SS>> {
        let forward_results = self.filter(initial_estimate, observations);
        self.smooth_from_filtered(forward_results)
    }

    /// Rauch-Tung-Striebel (RTS) smoother using already Kalman filtered estimates
    ///
    /// Operates on entire time series in one shot and returns a vector of state
    /// estimates. To be mathematically correct, the interval between
    /// observations must be the `dt` specified in the motion model.
    #[cfg(feature="std")]
    pub fn smooth_from_filtered(&self, mut forward_results: Vec<StateAndCovariance<R,SS>>) -> Vec<StateAndCovariance<R,SS>> {
        forward_results.reverse();

        let mut smoothed_backwards = Vec::with_capacity(forward_results.len());

        let mut smooth_future = forward_results[0].clone();
        smoothed_backwards.push(smooth_future.clone());
        for filt in forward_results.iter().skip(1) {
            smooth_future = self.smooth_step(&smooth_future,filt);
            smoothed_backwards.push(smooth_future.clone());
        }

        smoothed_backwards.reverse();
        smoothed_backwards
    }

    #[cfg(feature="std")]
    fn smooth_step(&self, smooth_future: &StateAndCovariance<R,SS>, filt: &StateAndCovariance<R,SS>) -> StateAndCovariance<R,SS> {
        let prior = self.transition_model.predict(filt);

        let v_chol = match na::linalg::Cholesky::new(prior.covariance().clone()) {
            Some(v) => v,
            None => {
                panic!("Covariance not positive semi-definite.");
            },
        };
        let inv_prior_covariance: MatrixMN<R,SS,SS> = v_chol.inverse();
        trace!("inv_prior_covariance {}", pretty_print!(inv_prior_covariance));

        // J = dot(Vfilt, dot(A.T, inv(Vpred)))  # smoother gain matrix
        let j = filt.covariance() * (self.transition_model.transition_model_transpose() * inv_prior_covariance);

        // xsmooth = xfilt + dot(J, xsmooth_future - xpred)
        let residuals = smooth_future.state() - prior.state();
        let state = filt.state() + &j*residuals;

        // Vsmooth = Vfilt + dot(J, dot(Vsmooth_future - Vpred, J.T))
        let covar_residuals = smooth_future.covariance() - prior.covariance();
        let covariance = filt.covariance() + &j * (covar_residuals * j.transpose());

        StateAndCovariance::new(state, covariance )
    }

}

#[inline]
fn is_nan<R: RealField>(x: R) -> bool {
    // Why isn't this implemented for RealField directly?
    let zero = na::convert(0.0);
    if x < zero {
        false
    } else {
        if x >= zero {
            false
        } else {
            true
        }
    }
}

#[test]
fn test_is_nan() {
    assert_eq!(is_nan::<f64>(-1.0), false);
    assert_eq!(is_nan::<f64>(0.0), false);
    assert_eq!(is_nan::<f64>(1.0), false);
    assert_eq!(is_nan::<f64>(1.0/0.0), false);
    assert_eq!(is_nan::<f64>(-1.0/0.0), false);
    assert_eq!(is_nan::<f64>(std::f64::NAN), true);

    assert_eq!(is_nan::<f32>(-1.0), false);
    assert_eq!(is_nan::<f32>(0.0), false);
    assert_eq!(is_nan::<f32>(1.0), false);
    assert_eq!(is_nan::<f32>(1.0/0.0), false);
    assert_eq!(is_nan::<f32>(-1.0/0.0), false);
    assert_eq!(is_nan::<f32>(std::f32::NAN), true);
}
