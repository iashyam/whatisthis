# Bug Report: OSError due to absolute artifact path preservation across environments

## Description
When running `mlflow.pytorch.log_model` on MacOS, the process crashes with `OSError: [Errno 45] Operation not supported: '/home/taylor'`. This appears to happen because the `mlruns` directory contains experiment metadata with an absolute path (`/home/taylor/...`) from a different environment (likely Linux), and MLflow attempts to use this absolute path as the artifact location on the new machine.

## detailed Traceback
```python
mlflow.pytorch.log_model(model, name="mildly complex model")
# ...
File "site-packages/mlflow/store/artifact/local_artifact_repo.py", line 69, in log_artifacts
    mkdir(artifact_dir)
File "site-packages/mlflow/utils/file_utils.py", line 207, in mkdir
    os.makedirs(target, exist_ok=True)
OSError: [Errno 45] Operation not supported: '/home/taylor'
```

## Steps to Reproduce
1. Create an MLflow experiment on a Linux machine (e.g., user `taylor`) where the default artifact location resolves to `/home/taylor/...`.
2. Copy the `mlruns` directory to a MacOS machine.
3. Run a Python script that uses `mlflow.set_experiment()` to select this existing experiment.
4. Attempt to log a model using `mlflow.pytorch.log_model()`.

## Environment
- **OS**: MacOS
- **Python Version**: 3.14
- **MLflow Version**: 3.8.1
- **PyTorch Version**: 2.9.1

## Possible Cause
The `meta.yaml` associated with the experiment in `mlruns` likely contains a hardcoded absolute path `artifact_location: file:///home/taylor/...`. MLflow's `LocalArtifactRepository` tries to ensure this directory exists, causing the crash on systems where that root path is invalid or unwritable.

## Suggested Fix
MLflow should detect if the `artifact_location` is an absolute path that does not exist or is invalid on the current system and either:
1. Warn the user and refuse to use the old path, or
2. Allow reconfiguring the artifact location relative to the current `mlruns` directory.
