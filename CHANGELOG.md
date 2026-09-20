# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.3.0

### Added

- Conditional Python and R formatter and linter configuration in generated `panache.toml` files
- Mustache templating for generated GitHub pre-commit action and workflow resources
- Conditional R cache configuration and `setup-r` workflow setup for R projects

### Changed

- Added reusable templated-file handling to the CLI resource tasks
- Preserved GitHub Actions expressions while rendering Mustache templates
- Rename from "actions" to "tasks"
- Move from RepoCLI to Run
- Tasks are not "done" vs. not; tasks are now done, ready, or blocked

### Removed

- Interactive option (instead allow for patching `._choose_repo_root()`, `._choose_features()`, `._choose_tasks()`)
