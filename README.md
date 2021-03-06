# adskalman-rs

[![Crates.io](https://img.shields.io/crates/v/adskalman.svg)](https://crates.io/crates/adskalman)
[![Documentation](https://docs.rs/adskalman/badge.svg)](https://docs.rs/adskalman/)
[![Crate License](https://img.shields.io/crates/l/adskalman.svg)](https://crates.io/crates/adskalman)
[![build](https://github.com/strawlab/adskalman-rs/workflows/build/badge.svg?branch=main)](https://github.com/strawlab/adskalman-rs/actions?query=branch%3Amain)

Kalman filter and Rauch-Tung-Striebel smoothing implementation.

* Includes various methods of computing the covariance matrix on the update step.
* Estimates state of arbitrary dimensions using observations of arbitrary dimension.
* Types are checked at compile time.
* Uses [nalgebra](https://nalgebra.org) for linear algebra.
* Supports `no_std` operation to run on embedded devices.

### disabling log::trace in release builds

To support debugging, `adskalman` extensively uses the `log::trace!()` macro.
You probably do not want this in your release builds. Therefore, in your
top-level application crate, you may want to use the `release_max_level_debug`
feature for the log crate like so:

```
[dependencies]
log = { version = "0.4", features = ["release_max_level_debug"] }
```

See [the `log` documentation](https://docs.rs/log/) for more information.

### Running the examples

There are several examples in the `examples/` directory, which is its own crate.
Run them like so:

```
cd examples
cargo run --bin online_tracking
```

## Code of conduct

Anyone who interacts with this software in any space, including but not limited
to this GitHub repository, must follow our [code of
conduct](code_of_conduct.md).

## License

Licensed under either of these:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/licenses/MIT)
