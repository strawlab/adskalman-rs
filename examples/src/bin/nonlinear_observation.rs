use na::allocator::Allocator;
use na::dimension::DimMin;
use na::dimension::{U1, U2, U4};
use na::DefaultAllocator;
use na::{MatrixMN, MatrixN, Vector2, Vector4, VectorN};
use nalgebra as na;
use nalgebra_rand_mvn::rand_mvn;

use adskalman::{KalmanFilterNoControl, ObservationModelLinear, TransitionModelLinearNoControl};
use adskalman_examples::motion_model;
use adskalman_examples::print_csv;

type MyType = f64;

// observation model -------

/// In this example, a 4D state and a 2D observation.
struct NonlinearObservationModel {}

impl NonlinearObservationModel {
    fn new() -> Self {
        Self {}
    }
    fn linearize_at(&self, state: &VectorN<MyType, U4>) -> Result<MyObservationModel, ()> {
        let evaluation_func = |state: &VectorN<MyType, U4>| {
            Some(VectorN::<MyType, U2>::new(
                state.x * state.x * state.x,
                state.x * state.y,
            ))
        };

        // Create observation model. We observe a nonlinear transformed position.
        #[rustfmt::skip]
        let observation_matrix = MatrixMN::<MyType, U2, U4>::new(
            3.0 * state.x * state.x, 0.0, 0.0, 0.0,
            state.y, state.x, 0.0, 0.0,
        );
        let observation_matrix_transpose = observation_matrix.transpose();
        let observation_noise_covariance = MatrixN::<MyType, U2>::new(0.01, 0.0, 0.0, 0.01);

        Ok(MyObservationModel {
            evaluation_func: Box::new(evaluation_func),
            observation_matrix,
            observation_matrix_transpose,
            observation_noise_covariance,
        })
    }
}

struct MyObservationModel
where
    DefaultAllocator: Allocator<MyType, U4, U4>,
    DefaultAllocator: Allocator<MyType, U2, U4>,
    DefaultAllocator: Allocator<MyType, U4, U2>,
    DefaultAllocator: Allocator<MyType, U2, U2>,
    DefaultAllocator: Allocator<MyType, U4>,
{
    evaluation_func: Box<dyn Fn(&VectorN<MyType, U4>) -> Option<VectorN<MyType, U2>>>,
    observation_matrix: MatrixMN<MyType, U2, U4>,
    observation_matrix_transpose: MatrixMN<MyType, U4, U2>,
    observation_noise_covariance: MatrixN<MyType, U2>,
}

impl ObservationModelLinear<MyType, U4, U2> for MyObservationModel
where
    DefaultAllocator: Allocator<MyType, U4, U4>,
    DefaultAllocator: Allocator<MyType, U2, U4>,
    DefaultAllocator: Allocator<MyType, U4, U2>,
    DefaultAllocator: Allocator<MyType, U2, U2>,
    DefaultAllocator: Allocator<MyType, U4>,
    DefaultAllocator: Allocator<MyType, U2>,
    DefaultAllocator: Allocator<(usize, usize), U2>,
    U2: DimMin<U2, Output = U2>,
{
    fn observation_matrix(&self) -> &MatrixMN<MyType, U2, U4> {
        &self.observation_matrix
    }
    fn observation_matrix_transpose(&self) -> &MatrixMN<MyType, U4, U2> {
        &self.observation_matrix_transpose
    }
    fn observation_noise_covariance(&self) -> &MatrixN<MyType, U2> {
        &self.observation_noise_covariance
    }
    fn evaluate(&self, state: &VectorN<MyType, U4>) -> Option<VectorN<MyType, U2>> {
        (*self.evaluation_func)(state)
    }
}

/// Given an observation model, compute a noisy observation for a state
fn compute_observation(
    observation_model_gen: &NonlinearObservationModel,
    state: &VectorN<MyType, U4>,
) -> Option<VectorN<MyType, U2>> {
    let observation_model = observation_model_gen.linearize_at(&state).unwrap();
    let noise_sample: MatrixMN<MyType, U1, U2> = rand_mvn(
        &Vector2::<MyType>::zeros(),
        observation_model.observation_noise_covariance,
    )
    .unwrap();
    match observation_model.evaluate(state) {
        Some(noise_free_obs) => Some(noise_free_obs + noise_sample.transpose()),
        None => None,
    }
}

// the main program --------

fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let dt = 0.01;
    let true_initial_state = VectorN::<MyType, U4>::new(0.0, 0.0, 10.0, -5.0);
    #[rustfmt::skip]
    let initial_covariance = MatrixN::<MyType, U4>::new(
        0.1, 0.0, 0.0, 0.0,
        0.0, 0.1, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.0,
        0.0, 0.0, 0.0, 0.1,
    );

    let motion_model = motion_model::ConstantVelocity2DModel::new(dt, 100.0);
    let observation_model_gen = NonlinearObservationModel::new();

    // Create some fake data with our model.
    let mut current_state = true_initial_state.clone();
    let mut state = vec![];
    let mut times = vec![];
    let zero4 = Vector4::<MyType>::zeros();
    let mut cur_time = 0.0;
    while cur_time < 0.5 {
        times.push(cur_time.clone());
        state.push(current_state.clone());
        let noise_sample: MatrixMN<MyType, U1, U4> =
            rand_mvn(&zero4, motion_model.transition_noise_covariance).unwrap();
        let noise_sample_col: VectorN<MyType, U4> = noise_sample.transpose();
        current_state = motion_model.transition_model * &current_state + noise_sample_col;
        cur_time += dt;
    }

    // Create noisy observations.
    let mut observations = vec![];
    for current_state in state.iter() {
        observations.push(compute_observation(&observation_model_gen, &current_state));
    }

    let mut previous_estimate =
        adskalman::StateAndCovariance::new(true_initial_state, initial_covariance);

    let mut state_estimates = vec![];
    for observation in observations.iter() {
        let this_estimate = if let (Some(observation), Some(observation_model)) = (
            observation.as_ref(),
            observation_model_gen
                .linearize_at(&previous_estimate.state())
                .ok(),
        ) {
            // We have an observation and an observation model.
            let kf = KalmanFilterNoControl::new(&motion_model, &observation_model);
            kf.step(&previous_estimate, observation)?
        } else {
            // Update the estimate with no observation.
            motion_model.predict(&previous_estimate)
        };

        state_estimates.push(this_estimate.state().clone());
        previous_estimate = this_estimate;
    }
    print_csv::print_csv_opt(&times, &state, &observations, &state_estimates);
    Ok(())
}
