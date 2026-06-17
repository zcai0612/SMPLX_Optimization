import json
import pickle

import numpy as np
import torch
import trimesh
from smplx import build_layer

from transfer_model.utils import batch_rodrigues


PARAM_OUTPUT_KEYS = {
    "transl",
    "global_orient",
    "body_pose",
    "betas",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
    "scale",
    "joints",
}


def _to_batch(vertices, device, dtype):
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices, dtype=dtype, device=device)
    else:
        vertices = vertices.to(device=device, dtype=dtype)
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(
            f"vertices must have shape [V, 3] or [B, V, 3], got {vertices.shape}"
        )
    return vertices


def _to_long_tensor(value, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.long)
    return torch.as_tensor(value, device=device, dtype=torch.long)


def _build_default_cfg(
    optim_type="trust-ncg",
    maxiters=100,
    ftol=-1.0,
    gtol=1e-8,
    interactive=False,
    summary_steps=5,
    edge_per_part=True,
):
    return {
        "summary_steps": summary_steps,
        "interactive": interactive,
        "optim": {
            "type": optim_type,
            "lr": 1.0,
            "gtol": gtol,
            "ftol": ftol,
            "maxiters": maxiters,
            "lbfgs": {
                "line_search_fn": "strong_wolfe",
                "max_iter": 50,
            },
            "sgd": {
                "momentum": 0.9,
                "nesterov": True,
            },
            "adam": {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False,
            },
            "trust_ncg": {
                "max_trust_radius": 1000,
                "initial_trust_radius": 0.05,
                "eta": 0.15,
                "gtol": 1e-5,
            },
        },
        "edge_fitting": {
            "per_part": edge_per_part,
            "reduction": "mean",
        },
        "vertex_fitting": {
            "type": "l2",
            "reduction": "mean",
        },
    }


def _module_device_dtype(module):
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    if tensor is None:
        return torch.device("cpu"), torch.float32
    return tensor.device, tensor.dtype


def _make_smplx_layer(
    model_path="human_models/models",
    device=None,
    num_betas=10,
    gender="neutral",
    use_pca=False,
    flat_hand_mean=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    return build_layer(
        model_path,
        model_type="smplx",
        gender=gender,
        num_betas=num_betas,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
    ).to(device=device)


def _load_smpl_to_smplx_correspondence(
    correspondence_path="human_models/smpl_data/smplx_to_smpl.pkl",
    device=None,
    dtype=torch.float32,
):
    with open(correspondence_path, "rb") as f:
        correspondence = pickle.load(f, encoding="latin1")

    missing_keys = {"closest_faces", "bc"}.difference(correspondence)
    if missing_keys:
        raise KeyError(
            f"SMPL to SMPL-X correspondence is missing keys: {sorted(missing_keys)}"
        )

    closest_faces = torch.as_tensor(
        correspondence["closest_faces"].astype(np.int64),
        device=device,
        dtype=torch.long,
    )
    bc = torch.as_tensor(
        correspondence["bc"].astype(np.float32),
        device=device,
        dtype=dtype,
    )
    if closest_faces.ndim != 2 or closest_faces.shape[1] != 3:
        raise ValueError(
            f"closest_faces must have shape [N, 3], got {closest_faces.shape}"
        )
    if bc.shape != closest_faces.shape:
        raise ValueError(f"bc must have shape {closest_faces.shape}, got {bc.shape}")
    return closest_faces, bc


def _load_faces(faces_path):
    faces = np.load(faces_path).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {faces.shape}")
    return faces


def _load_smplx_lmk_indices(smplx_lmk_indices_path, device=None):
    with open(smplx_lmk_indices_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "smplx_landmark_indices" not in data:
            raise KeyError(
                "Landmark index json must contain key 'smplx_landmark_indices'"
            )
        data = data["smplx_landmark_indices"]

    indices = torch.as_tensor(data, device=device, dtype=torch.long)
    if indices.ndim != 1:
        raise ValueError(
            f"smplx landmark indices must be a 1D list, got shape {indices.shape}"
        )
    return indices


def _slice_fitted_value(value, index):
    if torch.is_tensor(value):
        return value[index].detach().cpu()
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value[index] if value.ndim > 0 and value.shape[0] > index else value
    if isinstance(value, (list, tuple)):
        return value[index] if len(value) > index else value
    return value


def _build_fitted_outputs(var_dict):
    fitted_vertices = var_dict["vertices"].detach().cpu()
    fitted_faces = var_dict["faces"].astype(np.int64)
    fitted_meshes = [
        trimesh.Trimesh(
            vertices=fitted_vertices[i].numpy(),
            faces=fitted_faces,
            process=False,
        )
        for i in range(fitted_vertices.shape[0])
    ]

    fitted_params_list = []
    for i in range(fitted_vertices.shape[0]):
        fitted_params = {}
        for key, value in var_dict.items():
            if key in PARAM_OUTPUT_KEYS:
                fitted_params[key] = _slice_fitted_value(value, i)
        fitted_params_list.append(fitted_params)

    return fitted_meshes, fitted_params_list


def _scale_factor_from_var_dict(var_dict):
    scale = var_dict.get("scale")
    if scale is None:
        return None
    return torch.exp(scale).reshape(scale.shape[0], 1, 1)


def _apply_scale_to_body_output(body_model_output, var_dict):
    scale_factor = _scale_factor_from_var_dict(var_dict)
    if scale_factor is None:
        return body_model_output

    scaled_output = dict(body_model_output)
    if "vertices" in scaled_output:
        scaled_output["vertices"] = scaled_output["vertices"] * scale_factor
    if "joints" in scaled_output:
        scaled_output["joints"] = scaled_output["joints"] * scale_factor
    return scaled_output


def _sample_smplx_vertices_as_smpl(smplx_vertices, closest_faces, bc):
    face_vertices = smplx_vertices[:, closest_faces.reshape(-1)]
    face_vertices = face_vertices.reshape(
        smplx_vertices.shape[0],
        closest_faces.shape[0],
        3,
        3,
    )
    return torch.einsum("bvij,vi->bvj", face_vertices, bc)


def _build_param_dict(var_dict):
    param_dict = {}
    for key, var in var_dict.items():
        if key == "scale":
            continue
        if "pose" in key or "orient" in key:
            param_dict[key] = batch_rodrigues(var.reshape(-1, 3)).reshape(
                len(var), -1, 3, 3
            )
        else:
            param_dict[key] = var
    return param_dict


def _build_model_forward_closure_with_scale(
    body_model,
    var_dict,
    per_part=True,
    part_key=None,
    jidx=None,
    part=None,
):
    if per_part:
        assert part is not None and part_key is not None and jidx is not None

        def model_forward():
            param_dict = {}
            for key, var in var_dict.items():
                if key == "scale":
                    continue
                if part_key == key:
                    param_dict[key] = batch_rodrigues(var.reshape(-1, 3)).reshape(
                        len(var), -1, 3, 3
                    )
                    param_dict[key][:, jidx] = batch_rodrigues(
                        part.reshape(-1, 3)
                    ).reshape(-1, 3, 3)
                elif "pose" in key or "orient" in key:
                    param_dict[key] = batch_rodrigues(var.reshape(-1, 3)).reshape(
                        len(var), -1, 3, 3
                    )
                else:
                    param_dict[key] = var
            return _apply_scale_to_body_output(
                body_model(return_full_pose=True, get_skin=True, **param_dict),
                var_dict,
            )
    else:

        def model_forward():
            return _apply_scale_to_body_output(
                body_model(
                    return_full_pose=True,
                    get_skin=True,
                    **_build_param_dict(var_dict),
                ),
                var_dict,
            )

    return model_forward


def _finalize_var_dict(var_dict, body_model):
    body_model_output = body_model(return_full_pose=True, **_build_param_dict(var_dict))
    body_model_output = _apply_scale_to_body_output(body_model_output, var_dict)
    if "scale" in var_dict:
        var_dict["scale"] = torch.exp(var_dict["scale"])
    var_dict.update(body_model_output)
    var_dict["faces"] = body_model.faces
    return var_dict
