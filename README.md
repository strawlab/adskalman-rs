# adskalman-rs

Kalman filter implementation

* supports `no_std` operation to run on embedded devices
* general of state dimension and motion and observation models
* uses [nalgebra](https://nalgebra.org) for linear algebra

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
