use nalgebra as na;

use na::{
    allocator::Allocator,
    dimension::DimMin,
    dimension::{U2, U4},
    DefaultAllocator, Matrix1x2, Matrix1x4, Matrix2, Matrix2x4, Matrix4, Matrix4x2, OVector,
    Vector2, Vector4,
};
use nalgebra_rand_mvn::rand_mvn;

use adskalman::{KalmanFilterNoControl, ObservationModel};
use adskalman_examples::motion_model;
use adskalman_examples::print_csv;

type MyType = f64;

// observation model -------

/// This example has a 4D state and a 2D observation.
///
/// The observation is [x**3, xy].
struct NonlinearObservationModel {}

impl NonlinearObservationModel {
    /// Construct a new `NonlinearObservationModel`.
    fn new() -> Self {
        Self {}
    }
    /// Construct a new `LinearizedObservationModel` by linearizing around `state`.
    fn linearize_at(&self, state: &OVector<MyType, U4>) -> LinearizedObservationModel {
        let evaluation_func = |state: &OVector<MyType, U4>| {
            Vector2::<MyType>::new(state.x * state.x * state.x, state.x * state.y)
        };

        // Create Jacobian of the observation model. We only observe the position.
        #[rustfmt::skip]
        let observation_matrix = Matrix2x4::<MyType>::new(
            3.0 * state.x * state.x, 0.0, 0.0, 0.0,
            state.y, state.x, 0.0, 0.0,
        );
        let observation_matrix_transpose = observation_matrix.transpose();
        let observation_noise_covariance = Matrix2::<MyType>::new(0.01, 0.0, 0.0, 0.01);

        LinearizedObservationModel {
            evaluation_func: Box::new(evaluation_func),
            observation_matrix,
            observation_matrix_transpose,
            observation_noise_covariance,
        }
    }
}

type EvaluationFn = Box<dyn Fn(&Vector4<MyType>) -> Vector2<MyType>>;

struct LinearizedObservationModel
where
    DefaultAllocator: Allocator<MyType, U4, U4>,
    DefaultAllocator: Allocator<MyType, U2, U4>,
    DefaultAllocator: Allocator<MyType, U4, U2>,
    DefaultAllocator: Allocator<MyType, U2, U2>,
    DefaultAllocator: Allocator<MyType, U4>,
{
    evaluation_func: EvaluationFn,
    observation_matrix: Matrix2x4<MyType>,
    observation_matrix_transpose: Matrix4x2<MyType>,
    observation_noise_covariance: Matrix2<MyType>,
}

impl ObservationModel<MyType, U4, U2> for LinearizedObservationModel
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
    fn H(&self) -> &Matrix2x4<MyType> {
        &self.observation_matrix
    }
    fn HT(&self) -> &Matrix4x2<MyType> {
        &self.observation_matrix_transpose
    }
    fn R(&self) -> &Matrix2<MyType> {
        &self.observation_noise_covariance
    }
    fn predict_observation(&self, state: &Vector4<MyType>) -> Vector2<MyType> {
        (*self.evaluation_func)(state)
    }
}

// the main program --------

fn main() -> Result<(), anyhow::Error> {
    env_logger::init();

    let dt = 0.01;
    let true_initial_state = Vector4::<MyType>::new(0.0, 0.0, 10.0, -5.0);
    #[rustfmt::skip]
    let initial_covariance = Matrix4::<MyType>::new(
        0.1, 0.0, 0.0, 0.0,
        0.0, 0.1, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.0,
        0.0, 0.0, 0.0, 0.1,
    );

    let motion_model = motion_model::ConstantVelocity2DModel::new(dt, 100.0);
    let observation_model_gen = NonlinearObservationModel::new();

    // Create some fake data with our model.
    let mut current_state = true_initial_state;
    let mut state = vec![];
    let mut times = vec![];
    let zero4 = Vector4::<MyType>::zeros();
    let mut cur_time = 0.0;
    while cur_time < 0.5 {
        times.push(cur_time);
        state.push(current_state);
        let noise_sample: Matrix1x4<MyType> =
            rand_mvn(&zero4, motion_model.transition_noise_covariance).unwrap();
        let noise_sample_col: OVector<MyType, U4> = noise_sample.transpose();
        current_state = motion_model.transition_model * current_state + noise_sample_col;
        cur_time += dt;
    }

    // Create noisy observations.
    let mut observation = vec![];
    let zero2 = Vector2::<MyType>::zeros();
    for current_state in state.iter() {
        let observation_model = observation_model_gen.linearize_at(current_state);
        let noise_sample: Matrix1x2<MyType> =
            rand_mvn(&zero2, observation_model.observation_noise_covariance).unwrap();
        let noise_sample_col = noise_sample.transpose();
        let current_observation =
            observation_model.predict_observation(current_state) + noise_sample_col;
        observation.push(current_observation);
    }

    let mut previous_estimate =
        adskalman::StateAndCovariance::new(true_initial_state, initial_covariance);

    let mut state_estimates = vec![];
    for this_observation in observation.iter() {
        let observation_model = observation_model_gen.linearize_at(previous_estimate.state());
        let kf = KalmanFilterNoControl::new(&motion_model, &observation_model);

        let this_estimate = kf.step(&previous_estimate, this_observation)?;
        state_estimates.push(*this_estimate.state());
        previous_estimate = this_estimate;
    }
    print_csv::print_csv(&times, &state, &observation, &state_estimates);
    Ok(())
}
