use na::dimension::{U2, U4};
use na::RealField;
use na::VectorN;
use nalgebra as na;

// print to csv --------

pub fn print_csv<R: RealField>(
    times: &Vec<R>,
    state: &Vec<VectorN<R, U4>>,
    observation: &Vec<VectorN<R, U2>>,
    state_estimates: &Vec<VectorN<R, U4>>,
) {
    assert_eq!(times.len(), state.len());
    assert_eq!(times.len(), observation.len());
    assert_eq!(times.len(), state_estimates.len());
    println!("t,true_x,true_y,true_xvel,true_yvel,obs_x,obs_y,est_x,est_y,est_xvel,est_yvel");
    for i in 0..times.len() {
        #[rustfmt::skip]
        println!("{},{},{},{},{},{},{},{},{},{},{}",
            times[i], state[i][0], state[i][1], state[i][2], state[i][3],
            observation[i][0], observation[i][1],
            state_estimates[i][0], state_estimates[i][1], state_estimates[i][2], state_estimates[i][3],
        );
    }
}

pub fn print_csv_opt<R: RealField>(
    times: &Vec<R>,
    state: &Vec<VectorN<R, U4>>,
    opt_observation: &Vec<Option<VectorN<R, U2>>>,
    state_estimates: &Vec<VectorN<R, U4>>,
) {
    assert_eq!(times.len(), state.len());
    assert_eq!(times.len(), opt_observation.len());
    assert_eq!(times.len(), state_estimates.len());
    println!("t,true_x,true_y,true_xvel,true_yvel,obs_x,obs_y,est_x,est_y,est_xvel,est_yvel");
    for i in 0..times.len() {
        let obs = if let Some(obs) = opt_observation[i] {
            obs
        } else {
            VectorN::<R, U2>::new(na::convert(f64::NAN), na::convert(f64::NAN))
        };

        #[rustfmt::skip]
        println!("{},{},{},{},{},{},{},{},{},{},{}",
            times[i], state[i][0], state[i][1], state[i][2], state[i][3],
            obs[0], obs[1],
            state_estimates[i][0], state_estimates[i][1], state_estimates[i][2], state_estimates[i][3],
        );
    }
}
