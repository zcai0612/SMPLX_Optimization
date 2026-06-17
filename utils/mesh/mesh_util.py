import trimesh


def load_mesh_vertices(mesh_file):
    mesh = trimesh.load(mesh_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if len(geometries) != 1:
            raise ValueError(
                f"Expected one geometry in {mesh_file}, got {len(geometries)}"
            )
        mesh = geometries[0]
    return mesh.vertices


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    if with_normal or with_texture:
        raise NotImplementedError(
            "This lightweight loader only returns vertices and faces."
        )
    mesh = trimesh.load(mesh_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if len(geometries) != 1:
            raise ValueError(
                f"Expected one geometry in {mesh_file}, got {len(geometries)}"
            )
        mesh = geometries[0]
    return mesh.vertices, mesh.faces
