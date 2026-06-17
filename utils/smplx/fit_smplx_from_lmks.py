import torch

from transfer_model.losses import build_loss
from transfer_model.optimizers import build_optimizer, minimize
from transfer_model.transfer_model import get_variables

from utils.smplx.base import (
    _apply_scale_to_body_output,
    _build_default_cfg,
    _build_fitted_outputs,
    _build_model_forward_closure_with_scale,
    _build_param_dict,
    _finalize_var_dict,
    _load_smplx_lmk_indices,
    _make_smplx_layer,
    _module_device_dtype,
    _to_batch,
    _to_long_tensor,
)


def _summary_lmk_closure_with_scale(
    gt_landmarks,
    var_dict,
    body_model,
    smplx_lmk_indices,
    mask_ids=None,
):
    body_model_output = body_model(
        return_full_pose=True,
        get_skin=True,
        **_build_param_dict(var_dict),
    )
    est_landmarks = _apply_scale_to_body_output(body_model_output, var_dict)[
        "vertices"
    ][:, smplx_lmk_indices]
    if mask_ids is not None:
        est_landmarks = est_landmarks[:, mask_ids]
        gt_landmarks = gt_landmarks[:, mask_ids]

    lmk_error = (est_landmarks - gt_landmarks).pow(2).sum(dim=-1).sqrt().mean()
    return {"Landmark-to-SMPL-X": lmk_error * 1000}


def _build_lmk_vertex_closure_with_scale(
    body_model,
    var_dict,
    optimizer_dict,
    gt_landmarks,
    vertex_loss,
    smplx_lmk_indices,
    mask_ids=None,
    per_part=True,
    part_key=None,
    jidx=None,
    part=None,
    params_to_opt=None,
):
    optimizer = optimizer_dict["optimizer"]
    create_graph = optimizer_dict["create_graph"]
    model_forward = _build_model_forward_closure_with_scale(
        body_model,
        var_dict,
        per_part=per_part,
        part_key=part_key,
        jidx=jidx,
        part=part,
    )
    if params_to_opt is None:
        params_to_opt = [p for p in var_dict.values()]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()
        est_landmarks = model_forward()["vertices"][:, smplx_lmk_indices]
        loss = vertex_loss(
            est_landmarks[:, mask_ids] if mask_ids is not None else est_landmarks,
            gt_landmarks[:, mask_ids] if mask_ids is not None else gt_landmarks,
        )
        if backward:
            if create_graph:
                grads = torch.autograd.grad(loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                loss.backward()
        return loss

    return closure


def _run_smplx_lmk_fitting(
    exp_cfg,
    landmarks,
    body_model,
    smplx_lmk_indices,
    mask_ids=None,
    pred_scale=False,
):
    batch_size = len(landmarks)
    dtype, device = landmarks.dtype, landmarks.device
    summary_steps = exp_cfg.get("summary_steps")
    interactive = exp_cfg.get("interactive")
    show_progress = exp_cfg.get("show_progress", True)

    if landmarks.shape[1] != smplx_lmk_indices.shape[0]:
        raise ValueError(
            "landmark count does not match smplx_lmk_indices count: "
            f"{landmarks.shape[1]} vs {smplx_lmk_indices.shape[0]}"
        )

    var_dict = get_variables(batch_size, body_model, dtype=dtype)
    if pred_scale:
        var_dict["scale"] = torch.zeros(
            [batch_size, 1], device=device, dtype=dtype, requires_grad=True
        )
    optim_cfg = exp_cfg.get("optim", {})
    vertex_loss = build_loss(**exp_cfg.get("vertex_fitting", {})).to(device=device)

    def log_closure():
        return _summary_lmk_closure_with_scale(
            landmarks,
            var_dict,
            body_model,
            smplx_lmk_indices,
            mask_ids=mask_ids,
        )

    if "transl" in var_dict:
        params_to_opt = [var_dict["transl"]]
        if pred_scale:
            params_to_opt.append(var_dict["scale"])
        optimizer_dict = build_optimizer(params_to_opt, optim_cfg)
        closure = _build_lmk_vertex_closure_with_scale(
            body_model,
            var_dict,
            optimizer_dict,
            landmarks,
            vertex_loss=vertex_loss,
            smplx_lmk_indices=smplx_lmk_indices,
            mask_ids=mask_ids,
            per_part=False,
            params_to_opt=params_to_opt,
        )
        minimize(
            optimizer_dict["optimizer"],
            closure,
            params=params_to_opt,
            summary_closure=log_closure,
            summary_steps=summary_steps,
            interactive=interactive,
            show_progress=show_progress,
            **optim_cfg,
        )

    optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
    closure = _build_lmk_vertex_closure_with_scale(
        body_model,
        var_dict,
        optimizer_dict,
        landmarks,
        vertex_loss=vertex_loss,
        smplx_lmk_indices=smplx_lmk_indices,
        mask_ids=mask_ids,
        per_part=False,
    )
    minimize(
        optimizer_dict["optimizer"],
        closure,
        params=list(var_dict.values()),
        summary_closure=log_closure,
        summary_steps=summary_steps,
        interactive=interactive,
        show_progress=show_progress,
        **optim_cfg,
    )
    return _finalize_var_dict(var_dict, body_model)


class SMPLFitterLMK:
    def __init__(
        self,
        model_path="human_models/models",
        device=None,
        gender="neutral",
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        smplx_lmk_indices_path="data/smplx_600_landmark_253.json",
    ):
        self.body_model = _make_smplx_layer(
            model_path=model_path,
            device=device,
            num_betas=num_betas,
            gender=gender,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
        )
        self.device, self.dtype = _module_device_dtype(self.body_model)
        self.smplx_lmk_indices = _load_smplx_lmk_indices(
            smplx_lmk_indices_path,
            device=self.device,
        )

    def fit(
        self,
        landmarks,
        mask_ids=None,
        optim_type="trust-ncg",
        maxiters=100,
        ftol=-1.0,
        gtol=1e-8,
        interactive=False,
        summary_steps=5,
        edge_per_part=True,
        show_progress=False,
        pred_scale=False,
    ):
        landmark_vertices = _to_batch(landmarks, device=self.device, dtype=self.dtype)
        mask_ids = _to_long_tensor(mask_ids, device=self.device)
        exp_cfg = _build_default_cfg(
            optim_type=optim_type,
            maxiters=maxiters,
            ftol=ftol,
            gtol=gtol,
            interactive=interactive,
            summary_steps=summary_steps,
            edge_per_part=edge_per_part,
        )
        exp_cfg["show_progress"] = show_progress
        var_dict = _run_smplx_lmk_fitting(
            exp_cfg=exp_cfg,
            landmarks=landmark_vertices,
            body_model=self.body_model,
            smplx_lmk_indices=self.smplx_lmk_indices,
            mask_ids=mask_ids,
            pred_scale=pred_scale,
        )
        return _build_fitted_outputs(var_dict)


def fit_smplx_lmk(
    landmarks,
    smplx_lmk_indices_path="data/smplx_600_landmark_253.json",
    model_path="human_models/models",
    device=None,
    gender="neutral",
    mask_ids=None,
    optim_type="trust-ncg",
    maxiters=100,
    ftol=-1.0,
    gtol=1e-8,
    interactive=False,
    summary_steps=5,
    edge_per_part=True,
    num_betas=10,
    use_pca=False,
    flat_hand_mean=True,
    show_progress=False,
    pred_scale=False,
):
    fitter = SMPLFitterLMK(
        model_path=model_path,
        device=device,
        gender=gender,
        num_betas=num_betas,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        smplx_lmk_indices_path=smplx_lmk_indices_path,
    )
    return fitter.fit(
        landmarks=landmarks,
        mask_ids=mask_ids,
        optim_type=optim_type,
        maxiters=maxiters,
        ftol=ftol,
        gtol=gtol,
        interactive=interactive,
        summary_steps=summary_steps,
        edge_per_part=edge_per_part,
        show_progress=show_progress,
        pred_scale=pred_scale,
    )
