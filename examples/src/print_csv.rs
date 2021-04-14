use na::dimension::{U2, U4};
use na::OVector;
use na::RealField;
use nalgebra as na;

// print to csv --------

#[allow(dead_code)]
pub fn print_csv<R: RealField>(
    times: &[R],
    state: &[OVector<R, U4>],
    observation: &[OVector<R, U2>],
    state_estimates: &[OVector<R, U4>],
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
