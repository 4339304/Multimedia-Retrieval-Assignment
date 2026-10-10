"""
Step 1: minimal 3D mesh viewer for the Multimedia Retrieval assignment.

Usage:
    python view_mesh.py path/to/shape.obj [--wireframe]

Controls (Open3D window):
    - left-drag   : rotate
    - right-drag / scroll : zoom
    - ctrl+left-drag : pan
    - press 'w'   : toggle wireframe on top of shaded mesh
    - esc / close window to quit
"""

import argparse
import sys

import numpy as np
import open3d as o3d
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {path}")
    return mesh


def trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.65, 0.7, 0.8])
    return o3d_mesh


def main():
    parser = argparse.ArgumentParser(description="View a 3D shape (OBJ/OFF/PLY/STL).")
    parser.add_argument("shape_path", help="Path to the mesh file")
    parser.add_argument("--wireframe", action="store_true", help="Show edges overlaid on the shaded mesh")
    args = parser.parse_args()

    mesh = load_mesh(args.shape_path)
    print(f"Loaded: {args.shape_path}")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  faces:    {len(mesh.faces)}")
    print(f"  bounding box (min): {mesh.bounds[0]}")
    print(f"  bounding box (max): {mesh.bounds[1]}")

    o3d_mesh = trimesh_to_o3d(mesh)
    geometries = [o3d_mesh]

    if args.wireframe:
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
        wire.paint_uniform_color([0.1, 0.1, 0.1])
        geometries.append(wire)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=args.shape_path,
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    sys.exit(main())
