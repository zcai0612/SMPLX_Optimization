import argparse
import pickle as pkl

import trimesh

from utils.smplx.fit_smplx_from_lmks import SMPLFitterLMK


def load_mesh_vertices(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if len(geometries) != 1:
            raise ValueError(f"Expected one geometry in {path}, got {len(geometries)}")
        mesh = geometries[0]
    return mesh.vertices


def parse_args():
    parser = argparse.ArgumentParser(description="Fit SMPL-X from SMPL-X landmarks.")
    parser.add_argument("--landmarks", default="examples/pred_lmks.ply")
    parser.add_argument("--indices", default="data/smplx_600_landmark_253.json")
    parser.add_argument("--output-mesh", default="fitted_smplx_mesh.obj")
    parser.add_argument("--output-params", default="smplx_params.pkl")
    parser.add_argument("--model-path", default="human_models/models")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--optim-type", default="lbfgs")
    parser.add_argument("--maxiters", type=int, default=20)
    parser.add_argument("--pred-scale", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    landmarks = load_mesh_vertices(args.landmarks)

    fitter = SMPLFitterLMK(
        model_path=args.model_path,
        device=args.device,
        gender=args.gender,
        smplx_lmk_indices_path=args.indices,
    )
    fitted_meshes, fitted_params = fitter.fit(
        landmarks=landmarks,
        optim_type=args.optim_type,
        maxiters=args.maxiters,
        edge_per_part=False,
        show_progress=args.show_progress,
        pred_scale=args.pred_scale,
    )

    fitted_meshes[0].export(args.output_mesh)
    with open(args.output_params, "wb") as f:
        pkl.dump(fitted_params[0], f)


if __name__ == "__main__":
    main()
