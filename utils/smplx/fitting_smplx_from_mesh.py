import numpy as np
import torch

from transfer_model.losses import build_loss
from transfer_model.optimizers import build_optimizer, minimize
from transfer_model.transfer_model import get_variables
from transfer_model.utils import get_vertices_per_edge

from .base import (
    _apply_scale_to_body_output,
    _build_default_cfg,
    _build_fitted_outputs,
    _build_model_forward_closure_with_scale,
    _build_param_dict,
    _finalize_var_dict,
    _load_faces,
    _load_smpl_to_smplx_correspondence,
    _make_smplx_layer,
    _module_device_dtype,
    _sample_smplx_vertices_as_smpl,
    _to_batch,
    _to_long_tensor,
)


def _summary_closure_with_scale(gt_vertices, var_dict, body_model, mask_ids=None):
    body_model_output = body_model(
        return_full_pose=True,
        get_skin=True,
        **_build_param_dict(var_dict),
    )
    est_vertices = _apply_scale_to_body_output(body_model_output, var_dict)["vertices"]
    if mask_ids is not None:
        est_vertices = est_vertices[:, mask_ids]
        gt_vertices = gt_vertices[:, mask_ids]

    v2v = (est_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean()
    return {"Vertex-to-Vertex": v2v * 1000}


def _summary_corresponded_closure_with_scale(
    gt_vertices,
    var_dict,
    body_model,
    closest_faces,
    bc,
    mask_ids=None,
):
    body_model_output = body_model(
        return_full_pose=True,
        get_skin=True,
        **_build_param_dict(var_dict),
    )
    est_vertices = _apply_scale_to_body_output(body_model_output, var_dict)["vertices"]
    est_vertices = _sample_smplx_vertices_as_smpl(est_vertices, closest_faces, bc)
    if mask_ids is not None:
        est_vertices = est_vertices[:, mask_ids]
        gt_vertices = gt_vertices[:, mask_ids]

    v2v = (est_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean()
    return {"SMPL Vertex-to-Corresponded SMPL-X": v2v * 1000}


def _build_edge_closure_with_scale(
    body_model,
    var_dict,
    edge_loss,
    optimizer_dict,
    gt_vertices,
    per_part=True,
    part_key=None,
    jidx=None,
    part=None,
):
    optimizer = optimizer_dict["optimizer"]
    create_graph = optimizer_dict["create_graph"]
    if per_part:
        params_to_opt = [part]
        if "scale" in var_dict:
            params_to_opt.append(var_dict["scale"])
    else:
        params_to_opt = [
            p for key, p in var_dict.items() if "pose" in key or key == "scale"
        ]

    model_forward = _build_model_forward_closure_with_scale(
        body_model,
        var_dict,
        per_part=per_part,
        part_key=part_key,
        jidx=jidx,
        part=part,
    )

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()
        loss = edge_loss(model_forward()["vertices"], gt_vertices)
        if backward:
            if create_graph:
                grads = torch.autograd.grad(loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                loss.backward()
        return loss

    return closure


def _build_corresponded_edge_closure_with_scale(
    body_model,
    var_dict,
    edge_loss,
    optimizer_dict,
    gt_vertices,
    closest_faces,
    bc,
    per_part=True,
    part_key=None,
    jidx=None,
    part=None,
):
    optimizer = optimizer_dict["optimizer"]
    create_graph = optimizer_dict["create_graph"]
    if per_part:
        params_to_opt = [part]
        if "scale" in var_dict:
            params_to_opt.append(var_dict["scale"])
    else:
        params_to_opt = [
            p for key, p in var_dict.items() if "pose" in key or key == "scale"
        ]

    model_forward = _build_model_forward_closure_with_scale(
        body_model,
        var_dict,
        per_part=per_part,
        part_key=part_key,
        jidx=jidx,
        part=part,
    )

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()
        est_vertices = _sample_smplx_vertices_as_smpl(
            model_forward()["vertices"], closest_faces, bc
        )
        loss = edge_loss(est_vertices, gt_vertices)
        if backward:
            if create_graph:
                grads = torch.autograd.grad(loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                loss.backward()
        return loss

    return closure


def _build_vertex_closure_with_scale(
    body_model,
    var_dict,
    optimizer_dict,
    gt_vertices,
    vertex_loss,
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
        est_vertices = model_forward()["vertices"]
        loss = vertex_loss(
            est_vertices[:, mask_ids] if mask_ids is not None else est_vertices,
            gt_vertices[:, mask_ids] if mask_ids is not None else gt_vertices,
        )
        if backward:
            if create_graph:
                grads = torch.autograd.grad(loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                loss.backward()
        return loss

    return closure


def _build_corresponded_vertex_closure_with_scale(
    body_model,
    var_dict,
    optimizer_dict,
    gt_vertices,
    vertex_loss,
    closest_faces,
    bc,
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
        est_vertices = _sample_smplx_vertices_as_smpl(
            model_forward()["vertices"], closest_faces, bc
        )
        loss = vertex_loss(
            est_vertices[:, mask_ids] if mask_ids is not None else est_vertices,
            gt_vertices[:, mask_ids] if mask_ids is not None else gt_vertices,
        )
        if backward:
            if create_graph:
                grads = torch.autograd.grad(loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                loss.backward()
        return loss

    return closure


def _run_transfer_model_style_fitting(
    exp_cfg,
    vertices,
    body_model,
    mask_ids=None,
    pred_scale=False,
):
    batch_size = len(vertices)
    dtype, device = vertices.dtype, vertices.device
    summary_steps = exp_cfg.get("summary_steps")
    interactive = exp_cfg.get("interactive")
    show_progress = exp_cfg.get("show_progress", True)

    var_dict = get_variables(batch_size, body_model, dtype=dtype)
    if pred_scale:
        var_dict["scale"] = torch.zeros(
            [batch_size, 1], device=device, dtype=dtype, requires_grad=True
        )
    optim_cfg = exp_cfg.get("optim", {})

    if mask_ids is None:
        f_sel = np.ones_like(body_model.faces[:, 0], dtype=np.bool_)
    else:
        mask_ids_list = mask_ids.detach().cpu().numpy().tolist()
        f_per_v = [[] for _ in range(body_model.get_num_verts())]
        [f_per_v[vv].append(iff) for iff, ff in enumerate(body_model.faces) for vv in ff]
        f_sel = list(set(tuple(sum([f_per_v[vv] for vv in mask_ids_list], []))))

    vpe = get_vertices_per_edge(
        body_model.v_template.detach().cpu().numpy(),
        body_model.faces[f_sel],
    )

    def log_closure():
        return _summary_closure_with_scale(
            vertices, var_dict, body_model, mask_ids=mask_ids
        )

    edge_fitting_cfg = exp_cfg.get("edge_fitting", {})
    edge_loss = build_loss(
        type="vertex-edge",
        gt_edges=vpe,
        est_edges=vpe,
        **edge_fitting_cfg,
    ).to(device=device)

    vertex_loss = build_loss(**exp_cfg.get("vertex_fitting", {})).to(device=device)
    _run_edge_stage(
        body_model,
        var_dict,
        optim_cfg,
        edge_fitting_cfg,
        edge_loss,
        vertices,
        log_closure,
        summary_steps,
        interactive,
        show_progress,
        pred_scale,
    )
    _run_vertex_stages(
        body_model,
        var_dict,
        optim_cfg,
        vertex_loss,
        vertices,
        log_closure,
        summary_steps,
        interactive,
        show_progress,
        pred_scale,
        mask_ids=mask_ids,
    )
    return _finalize_var_dict(var_dict, body_model)


def _run_smpl_to_smplx_fitting(
    exp_cfg,
    smpl_vertices,
    body_model,
    smpl_faces,
    closest_faces,
    bc,
    mask_ids=None,
    pred_scale=False,
):
    batch_size = len(smpl_vertices)
    dtype, device = smpl_vertices.dtype, smpl_vertices.device
    summary_steps = exp_cfg.get("summary_steps")
    interactive = exp_cfg.get("interactive")
    show_progress = exp_cfg.get("show_progress", True)

    if smpl_vertices.shape[1] != closest_faces.shape[0]:
        raise ValueError(
            "smpl_mesh vertex count does not match correspondence count: "
            f"{smpl_vertices.shape[1]} vs {closest_faces.shape[0]}"
        )

    var_dict = get_variables(batch_size, body_model, dtype=dtype)
    if pred_scale:
        var_dict["scale"] = torch.zeros(
            [batch_size, 1], device=device, dtype=dtype, requires_grad=True
        )
    optim_cfg = exp_cfg.get("optim", {})

    if mask_ids is None:
        f_sel = np.ones_like(smpl_faces[:, 0], dtype=np.bool_)
    else:
        mask_ids_list = mask_ids.detach().cpu().numpy().tolist()
        f_per_v = [[] for _ in range(smpl_vertices.shape[1])]
        [f_per_v[vv].append(iff) for iff, ff in enumerate(smpl_faces) for vv in ff]
        f_sel = list(set(tuple(sum([f_per_v[vv] for vv in mask_ids_list], []))))

    vpe = get_vertices_per_edge(
        smpl_vertices[0].detach().cpu().numpy(),
        smpl_faces[f_sel],
    )

    def log_closure():
        return _summary_corresponded_closure_with_scale(
            smpl_vertices,
            var_dict,
            body_model,
            closest_faces,
            bc,
            mask_ids=mask_ids,
        )

    edge_fitting_cfg = exp_cfg.get("edge_fitting", {})
    edge_loss = build_loss(
        type="vertex-edge",
        gt_edges=vpe,
        est_edges=vpe,
        **edge_fitting_cfg,
    ).to(device=device)

    vertex_loss = build_loss(**exp_cfg.get("vertex_fitting", {})).to(device=device)
    _run_edge_stage(
        body_model,
        var_dict,
        optim_cfg,
        edge_fitting_cfg,
        edge_loss,
        smpl_vertices,
        log_closure,
        summary_steps,
        interactive,
        show_progress,
        pred_scale,
        closest_faces=closest_faces,
        bc=bc,
    )
    _run_vertex_stages(
        body_model,
        var_dict,
        optim_cfg,
        vertex_loss,
        smpl_vertices,
        log_closure,
        summary_steps,
        interactive,
        show_progress,
        pred_scale,
        mask_ids=mask_ids,
        closest_faces=closest_faces,
        bc=bc,
    )
    return _finalize_var_dict(var_dict, body_model)


def _run_edge_stage(
    body_model,
    var_dict,
    optim_cfg,
    edge_fitting_cfg,
    edge_loss,
    vertices,
    log_closure,
    summary_steps,
    interactive,
    show_progress,
    pred_scale,
    closest_faces=None,
    bc=None,
):
    per_part = edge_fitting_cfg.get("per_part", True)
    edge_closure_builder = (
        _build_edge_closure_with_scale
        if closest_faces is None
        else _build_corresponded_edge_closure_with_scale
    )

    if per_part:
        for key, var in var_dict.items():
            if "pose" not in key:
                continue
            for jidx in range(var.shape[1]):
                part = torch.zeros(
                    [len(vertices), 3],
                    dtype=vertices.dtype,
                    device=vertices.device,
                    requires_grad=True,
                )
                params_to_opt = [part]
                if pred_scale:
                    params_to_opt.append(var_dict["scale"])
                optimizer_dict = build_optimizer(params_to_opt, optim_cfg)
                closure_kwargs = _correspondence_kwargs(closest_faces, bc)
                closure = edge_closure_builder(
                    body_model,
                    var_dict,
                    edge_loss,
                    optimizer_dict,
                    vertices,
                    per_part=True,
                    part_key=key,
                    jidx=jidx,
                    part=part,
                    **closure_kwargs,
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
                with torch.no_grad():
                    var[:, jidx] = part
        return

    optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
    closure = edge_closure_builder(
        body_model,
        var_dict,
        edge_loss,
        optimizer_dict,
        vertices,
        per_part=False,
        **_correspondence_kwargs(closest_faces, bc),
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


def _run_vertex_stages(
    body_model,
    var_dict,
    optim_cfg,
    vertex_loss,
    vertices,
    log_closure,
    summary_steps,
    interactive,
    show_progress,
    pred_scale,
    mask_ids=None,
    closest_faces=None,
    bc=None,
):
    vertex_closure_builder = (
        _build_vertex_closure_with_scale
        if closest_faces is None
        else _build_corresponded_vertex_closure_with_scale
    )

    if "transl" in var_dict:
        params_to_opt = [var_dict["transl"]]
        if pred_scale:
            params_to_opt.append(var_dict["scale"])
        optimizer_dict = build_optimizer(params_to_opt, optim_cfg)
        closure = vertex_closure_builder(
            body_model,
            var_dict,
            optimizer_dict,
            vertices,
            vertex_loss=vertex_loss,
            mask_ids=mask_ids,
            per_part=False,
            params_to_opt=params_to_opt,
            **_correspondence_kwargs(closest_faces, bc),
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
    closure = vertex_closure_builder(
        body_model,
        var_dict,
        optimizer_dict,
        vertices,
        vertex_loss=vertex_loss,
        mask_ids=mask_ids,
        per_part=False,
        **_correspondence_kwargs(closest_faces, bc),
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


def _correspondence_kwargs(closest_faces, bc):
    if closest_faces is None:
        return {}
    return {"closest_faces": closest_faces, "bc": bc}


class SMPLXFitter:
    def __init__(
        self,
        model_path="human_models/models",
        device=None,
        gender="neutral",
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
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

    def fit(
        self,
        smplx_mesh,
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
        vertices = _to_batch(smplx_mesh, device=self.device, dtype=self.dtype)
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
        var_dict = _run_transfer_model_style_fitting(
            exp_cfg=exp_cfg,
            vertices=vertices,
            body_model=self.body_model,
            mask_ids=mask_ids,
            pred_scale=pred_scale,
        )
        return _build_fitted_outputs(var_dict)


class SMPLXFromSMPLFitter:
    def __init__(
        self,
        model_path="human_models/models",
        device=None,
        gender="neutral",
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        correspondence_path="human_models/smpl_data/smplx_to_smpl.pkl",
        smpl_faces_path="human_models/smpl_data/smpl_faces.npy",
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
        self.closest_faces, self.bc = _load_smpl_to_smplx_correspondence(
            correspondence_path=correspondence_path,
            device=self.device,
            dtype=self.dtype,
        )
        self.smpl_faces = _load_faces(smpl_faces_path)

    def fit(
        self,
        smpl_mesh,
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
        vertices = _to_batch(smpl_mesh, device=self.device, dtype=self.dtype)
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
        var_dict = _run_smpl_to_smplx_fitting(
            exp_cfg=exp_cfg,
            smpl_vertices=vertices,
            body_model=self.body_model,
            smpl_faces=self.smpl_faces,
            closest_faces=self.closest_faces,
            bc=self.bc,
            mask_ids=mask_ids,
            pred_scale=pred_scale,
        )
        return _build_fitted_outputs(var_dict)


def fit_smplx(
    smplx_mesh,
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
    fitter = SMPLXFitter(
        model_path=model_path,
        device=device,
        gender=gender,
        num_betas=num_betas,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
    )
    return fitter.fit(
        smplx_mesh=smplx_mesh,
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


def fit_smpl_to_smplx(
    smpl_mesh,
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
    correspondence_path="human_models/smpl_data/smplx_to_smpl.pkl",
    smpl_faces_path="human_models/smpl_data/smpl_faces.npy",
):
    fitter = SMPLXFromSMPLFitter(
        model_path=model_path,
        device=device,
        gender=gender,
        num_betas=num_betas,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        correspondence_path=correspondence_path,
        smpl_faces_path=smpl_faces_path,
    )
    return fitter.fit(
        smpl_mesh=smpl_mesh,
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


# Compatibility for older imports:
from .fit_smplx_from_lmks import SMPLFitterLMK, fit_smplx_lmk  # noqa: E402
