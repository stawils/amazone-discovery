# Refactor Plan: Modular Pipeline & Provider Abstraction

## Objective
Refactor the Amazon Archaeological Discovery Pipeline to:
- Make each pipeline step (download, analyze, score, report, visualize) a modular, composable unit.
- Abstract data providers (USGS, GEE, etc.) behind a common interface.
- Standardize data objects passed between steps.
- Ensure the app remains functional and testable at every stage.

---

## Guiding Principles
- **Incremental changes:** Refactor in small, testable steps.
- **Backwards compatibility:** Keep the CLI and main pipeline working after each step.
- **Comprehensive testing:** Add/maintain tests for each refactor stage.
- **Documentation:** Update docstrings and README as interfaces change.

---

## Step-by-Step Plan

### 1. Preparation
- [x] Review and document current data flow between steps (download → analyze → score → report → visualize).
  - **Done:** Steps communicate via files and a results dict, not direct object passing. No standardized data object; provider output format varies.
- [x] Identify all places where provider-specific logic exists in the pipeline.
  - **Done:** Provider logic is present in main.py (pipeline class, CLI), src/gee_provider.py, src/usgs_api.py, and in the file structure/manifest assumptions.
- [x] Add/verify tests for current pipeline (smoke test, CLI test, synthetic data test).
  - **Done:** Synthetic tests (test_pipeline.py, simple_test.py, quick_test_setup.py) cover detection, scoring, and visualization using synthetic data. The testing guide describes how to run the real pipeline via CLI, but there are no automated CLI/integration tests for main.py. Recommend adding at least one automated CLI test for main.py as a future improvement.

### 2. Define Standard Data Objects
- [x] Create a `SceneData` class (in `src/data_objects.py` or similar) to encapsulate all info about a downloaded scene (zone, provider, file paths, available bands/features, metadata).
- [x] Update download methods (USGS, GEE) to return `SceneData` objects (in addition to current outputs, for compatibility).
- [x] Update analyze step to accept a list of `SceneData` objects (but keep old interface working).
  - **Done:** `analyze_downloaded_data` now accepts `scene_data_list` and processes scenes directly if provided, maintaining backward compatibility.
- [ ] Add tests for `SceneData` creation and usage.

### 3. Abstract Provider Interface
- [x] Define a `BaseProvider` abstract class with a method like `download_data(zones, max_scenes) -> List[SceneData]`.
- [x] Refactor USGS and GEE logic into `USGSProvider` and `GEEProvider` classes implementing this interface.
- [x] Update pipeline to instantiate provider via this interface, but keep old logic as fallback.
  - **Done:** Provider abstraction layer implemented. Pipeline now uses USGSProvider and GEEProvider via BaseProvider interface, with fallback to legacy logic.
- [ ] Add tests for provider selection and data download.

### 4. Modularize Pipeline Steps
- [x] Refactor each pipeline step (download, analyze, score, report, visualize) into its own class/module with a `run(input)` method.
- [x] Each step should accept and return standardized data objects.
- [x] Update the main pipeline to orchestrate these steps, passing outputs as inputs.
  - **Done:** All major pipeline steps are now modularized and orchestrated via ModularPipeline. Each step is a class with a run method, passing standardized data objects/results.
- [ ] Keep the current monolithic pipeline as a fallback until new pipeline is stable.
- [ ] Add tests for each step as a standalone module.

### 5. Feature Awareness & Adaptation
- [ ] Add feature/band availability flags to `SceneData`.
- [ ] Update analysis and scoring steps to check for required features and adapt/skip gracefully if missing.
- [ ] Add tests for feature-missing scenarios.

### 6. CLI and Backwards Compatibility
- [ ] Update CLI to use the new modular pipeline, but keep old CLI options working.
- [ ] Ensure all CLI commands (`--download`, `--analyze`, `--full-pipeline`, etc.) work as before.

### 7. Documentation & Cleanup
- [ ] Update README and docstrings to reflect new architecture and usage.
- [ ] Remove deprecated/legacy code after confirming new pipeline is stable.
- [ ] Final code cleanup and formatting.

---

## Milestones & Checkpoints
- After each major step, run all tests and verify CLI/manual workflows.
- Use feature branches for each refactor stage; merge to main only after passing tests.
- Maintain a changelog of interface changes and migration notes.

---

## Rollback Plan
- At any point, revert to the previous working version if a refactor step breaks core functionality.
- Keep the old pipeline logic until the new modular pipeline is fully validated.

---

## Notes
- Prioritize minimal disruption to users and existing workflows.
- Communicate interface changes clearly in documentation and commit messages.
- Encourage incremental PRs and code reviews for each step. 