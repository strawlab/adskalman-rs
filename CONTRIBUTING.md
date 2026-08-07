# Contributing

## Commit messages

This repository uses [Conventional Commits](https://www.conventionalcommits.org/)
to choose release versions and generate `CHANGELOG.md` with git-cliff.

- `feat:` adds user-visible functionality.
- `fix:` corrects user-visible behavior.
- `perf:` improves performance.
- `docs:` changes packaged documentation.
- `refactor:` changes packaged code without changing behavior.
- `chore(deps):` updates a dependency.
- `type!:` or a `BREAKING CHANGE:` footer marks an incompatible API change.
- `ci:`, `test:`, `style:`, `build:`, and other `chore:` commits do not trigger a release.

An optional scope may identify the affected area. Keep the subject imperative,
lowercase, and free of a trailing period. When using squash merge, make the pull
request title follow the same format because it becomes the commit subject.

## Release process

Pushing a releasable commit to `main` creates or updates a release PR. Merging that
PR publishes `adskalman` to crates.io, creates a version tag, and creates a GitHub
release. Do not edit package versions or released changelog entries by hand.

Configure crate `adskalman` on crates.io with a trusted publisher for repository
`strawlab/adskalman-rs`, workflow `release-plz.yml`, and environment `release`.
Restrict the GitHub `release` environment to the `main` branch. In the repository's
Actions settings, allow GitHub Actions to create pull requests. A repository secret
named `RELEASE_PLZ_TOKEN` is optional; configure it as a fine-grained GitHub token
with Contents and Pull requests write access if CI should run automatically on the
release PR. Without it, the workflow falls back to `GITHUB_TOKEN`.
