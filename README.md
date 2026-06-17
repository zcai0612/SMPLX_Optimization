# SMPL-X Fitting

This project fits SMPL-X model parameters from either a full SMPL-X mesh or a sparse set of SMPL-X landmark vertices.

## Features

- Fit SMPL-X parameters from a SMPL-X topology mesh.
- Fit SMPL-X parameters from landmarks sampled at known SMPL-X vertex indices.
- Optionally fit SMPL-X from SMPL topology vertices using the provided SMPL-to-SMPL-X correspondence.
- Export fitted SMPL-X meshes and parameter dictionaries.

The main APIs are:

```python
from utils.smplx import SMPLXFitter, SMPLFitterLMK
```

## Project Layout

```text
.
├── data/
│   └── smplx_600_landmark_253.json
├── examples/
│   └── pred_lmks.ply
├── human_models/
│   ├── models/
│   └── smpl_data/
├── transfer_model/
├── utils/
│   └── smplx/
│       ├── base.py
│       ├── fitting_smplx_from_mesh.py
│       └── fit_smplx_from_lmks.py
└── fit_lmk_example.py
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version if the default pip package is not suitable:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The SMPL-X model files must exist under:

```text
human_models/models/smplx/
```

Expected files include `SMPLX_NEUTRAL.pkl`, and optionally gender-specific `.pkl` / `.npz` files.
The full `human_models` directory can be downloaded from:

```text
https://huggingface.co/Co2y/UP2You/tree/main/human_models
```

For example:

```bash
export HF_ENDPOINT="https://hf-mirror.com"  # Optional
hf download Co2y/UP2You --include "human_models/**" --local-dir ./src
mv ./src/human_models ./
rm -rf ./src
```

## Usage

### Fit From Landmarks

The landmark fitter expects input landmarks with shape `[N, 3]` or `[B, N, 3]`, where `N` matches the number of vertex indices in the landmark index JSON.

Run the provided example:

```bash
python fit_lmk_example.py \
  --landmarks examples/pred_lmks.ply \
  --indices data/smplx_600_landmark_253.json \
  --optim-type lbfgs \
  --maxiters 20 \
  --output-mesh fitted_smplx_mesh.obj \
  --output-params smplx_params.pkl
```

Python API:

```python
from utils.mesh.mesh_util import load_mesh_vertices
from utils.smplx.fit_smplx_from_lmks import SMPLFitterLMK

landmarks = load_mesh_vertices("examples/pred_lmks.ply")

fitter = SMPLFitterLMK(
    model_path="human_models/models",
    smplx_lmk_indices_path="data/smplx_600_landmark_253.json",
    device="cuda:0",
)
meshes, params = fitter.fit(
    landmarks=landmarks,
    optim_type="lbfgs",
    maxiters=20,
    edge_per_part=False,
)
```

### Fit From SMPL-X Mesh

The mesh fitter expects SMPL-X vertices with shape `[10475, 3]` or `[B, 10475, 3]`.

```python
from utils.mesh.mesh_util import load_obj_mesh
from utils.smplx.fitting_smplx_from_mesh import SMPLXFitter

vertices, _ = load_obj_mesh("input_smplx_mesh.obj")

fitter = SMPLXFitter(
    model_path="human_models/models",
    device="cuda:0",
    gender="neutral",
)
meshes, params = fitter.fit(
    smplx_mesh=vertices,
    optim_type="lbfgs",
    maxiters=20,
    edge_per_part=False,
)
```

### Fit SMPL-X From SMPL Vertices

```python
from utils.smplx.fitting_smplx_from_mesh import SMPLXFromSMPLFitter

fitter = SMPLXFromSMPLFitter(
    model_path="human_models/models",
    correspondence_path="human_models/smpl_data/smplx_to_smpl.pkl",
    smpl_faces_path="human_models/smpl_data/smpl_faces.npy",
    device="cuda:0",
)
meshes, params = fitter.fit(
    smpl_mesh=smpl_vertices,
    optim_type="lbfgs",
    maxiters=20,
    edge_per_part=False,
)
```

## Outputs

Each fitting call returns:

- `meshes`: a list of `trimesh.Trimesh` fitted SMPL-X meshes.
- `params`: a list of dictionaries containing fitted SMPL-X parameters such as `transl`, `global_orient`, `body_pose`, `betas`, hand pose, face pose, expression, joints, and optional `scale`.

## Notes

- `lbfgs` is the recommended optimizer for the included examples.
- `trust-ncg` requires the optional `torchtrustncg` package.
- `pred_scale=True` enables optimization of a global scale factor.
- Use `mask_ids` to fit only selected vertices or landmarks.
