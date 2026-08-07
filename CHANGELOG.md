# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.0](https://github.com/strawlab/adskalman-rs/compare/0.17.0...0.18.0) - 2026-08-07

### Added
- *(deps)* [**breaking**] update nalgebra to 0.35

### Fixed
- *(release)* align examples publishing config

## [0.17.0] - 2025-08-01

[0.17.0]: https://github.com/strawlab/adskalman-rs/releases/tag/0.17.0

# 0.10.0 (June 4, 2021)

The `nalgebra` and `approx` crates have been updated to their latest versions.

# 0.9.0 (May 15, 2021)

### Changed

- Change the default covariance update method to use the Joseph form. This is
  more numerically robust at the cost of being more computationally expensive.
  To use the old covariance update method, call
  `KalmanFilterNoControl::step_with_options()` with argument
  `covariance_update_method` set to
  `CovarianceUpdateMethod::OptimalKalmanForcedSymmetric`.
- Fixed the spelling of `CovarianceUpdateMethod` (which was previously
  misspelled).

### Added

- Implemented tests to run basic sanity checks for the Kalman filter (with
  different covariance update methods), Kalman smoother, and the Kalman smoother
  with missing observations.

# 0.8.0 (April 14, 2021)

### Changed

- Change primary names of many functions to match conventional Kalman filter
  documentations (i.e. F for state transition model, H for observation model).
  Mostly old names are still kept but are deprecated. For implementations of the
  `TransitionModelLinearNoControl` trait, change `transition_model` to `F`,
  `transition_model_transpose` to `FT`, and `transition_noise_covariance` to
  `Q`. The trait name `ObservationModelLinear` was changed to `ObservationModel`
  and implementations of this trait need to change `observation_matrix` to `H`,
  `observation_matrix_transpose` to `HT`, `observation_noise_covariance` to `R`,
  and `evaluate` to `predict_observation`.

### Fixed

- examples do not raise clippy warnings ([#6])

[#6]: https://github.com/strawlab/adskalman-rs/pull/6

# 0.7.0 (April 13, 2021)

### Fixes

- Depend on nalgebra 0.26, which added initial support for const generics.
