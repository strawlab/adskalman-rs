use na::dimension::{U1, U2, U4};
use na::{OMatrix, OVector, Vector2, Vector4};
use nalgebra as na;
use nalgebra_rand_mvn::rand_mvn;

use adskalman::{KalmanFilterNoControl, ObservationModelLinear};

use adskalman_examples::linear_observation_model;
use adskalman_examples::motion_model;
use adskalman_examples::print_csv;

type MyType = f64;

// the main program --------

fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let dt = 0.01;
    let true_initial_state = OVector::<MyType, U4>::new(0.0, 0.0, 10.0, -5.0);
    #[rustfmt::skip]
    let initial_covariance = OMatrix::<MyType,U4,U4>::new(0.1, 0.0, 0.0, 0.0,
        0.0, 0.1, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.0,
        0.0, 0.0, 0.0, 0.1);

    let motion_model = motion_model::ConstantVelocity2DModel::new(dt, 100.0);
    let observation_model = linear_observation_model::PositionObservationModel::new(0.01);
    let kf = KalmanFilterNoControl::new(&motion_model, &observation_model);

    // Create some fake data with our model.
    let mut current_state = true_initial_state.clone();
    let mut state = vec![];
    let mut times = vec![];
    let zero4 = Vector4::<MyType>::zeros();
    let mut cur_time = 0.0;
    while cur_time < 0.5 {
        times.push(cur_time.clone());
        state.push(current_state.clone());
        let noise_sample: OMatrix<MyType, U1, U4> =
            rand_mvn(&zero4, motion_model.transition_noise_covariance).unwrap();
        let noise_sample_col: OVector<MyType, U4> = noise_sample.transpose();
        current_state = motion_model.transition_model * &current_state + noise_sample_col;
        cur_time += dt;
    }

    // Create noisy observations.
    let mut observation = vec![];
    let zero2 = Vector2::<MyType>::zeros();
    for current_state in state.iter() {
        let noise_sample: OMatrix<MyType, U1, U2> =
            rand_mvn(&zero2, observation_model.observation_noise_covariance).unwrap();
        let noise_sample_col = noise_sample.transpose();
        let current_observation = observation_model.evaluate(current_state) + noise_sample_col;
        observation.push(current_observation);
    }

    let mut previous_estimate =
        adskalman::StateAndCovariance::new(true_initial_state, initial_covariance);

    let mut state_estimates = vec![];
    for this_observation in observation.iter() {
        let this_estimate = kf.step(&previous_estimate, this_observation)?;
        state_estimates.push(this_estimate.state().clone());
        previous_estimate = this_estimate;
    }
    print_csv::print_csv(&times, &state, &observation, &state_estimates);
    Ok(())
}
