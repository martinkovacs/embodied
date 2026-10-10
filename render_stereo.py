"""Render synthetic stereo-vision test data with two cameras.

Usage:
    blender -b -P render_stereo.py -- [--out synthetic] [--n-calib 25] [--square-mm 25]

Outputs (in --out):
    calib/left_XX.png, calib/right_XX.png   checkerboard at random poses
    table_left.png / table_right.png        board lying flat on the table
    scene_left.png / scene_right.png        cylinders on the table
    ground_truth.json                       K, stereo R/T, board poses, cylinders

All internal units are metres; JSON distances/matrices are in millimetres.
Camera/geometry conventions: Blender camera looks down -Z with +Y up;
OpenCV camera looks down +Z with +Y down, so
    world->cv_cam rotation = diag(1, -1, -1) @ (blender world->cam rotation).
"""
import argparse
import math
import os
import random
import sys
from mathutils import Matrix, Vector

import bpy
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="synthetic")
ap.add_argument("--n-calib", type=int, default=25)
ap.add_argument("--square-mm", type=float, default=25.0)
args = ap.parse_args(argv)

OUT = os.path.abspath(args.out)
SQ = args.square_mm / 1000.0          # square size, metres
NX, NY = 10, 7                        # squares; 9x6 INNER corners
MARGIN = 1                            # white margin, in squares
BASE_W = (NX + 2 * MARGIN) * SQ       # white base plane (also sight-check box)
BASE_H = (NY + 2 * MARGIN) * SQ
BOARD_THICK = 0.0005                  # base plane sits 0.5 mm above its origin

CAM_H = 0.600                         # camera height above table
BASELINE = 0.150
CAM_Y = -0.280                        # horizontal offset; tilt ~ atan(0.28/0.60) ~ 25 deg
SEED = 12345
LENS_MM, SENSOR_MM = 35.0, 36.0

COLORS = {
    "red":   (0.80, 0.03, 0.02, 1),
    "blue":  (0.03, 0.06, 0.60, 1),
    "green": (0.15, 0.55, 0.20, 1),
    "white": (0.85, 0.85, 0.85, 1),
}
CYL_R, CYL_H = 0.0175, 0.030


def material(name, rgba, roughness):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    return m


class Board:
    """Checkerboard: thin white base plane + 35 black square quads (real geometry).

    Local frame: origin at board centre, x across the 10-square direction,
    y across the 7-square direction, z up out of the board.
    """

    def __init__(self, parent_collection):
        white = material("board_white", (0.90, 0.90, 0.90, 1), 0.5)
        black = material("board_black", (0.02, 0.02, 0.02, 1), 0.5)

        me = bpy.data.meshes.new("board")
        me.from_pydata(
            [(-BASE_W / 2, -BASE_H / 2, 0), (BASE_W / 2, -BASE_H / 2, 0),
             (BASE_W / 2, BASE_H / 2, 0), (-BASE_W / 2, BASE_H / 2, 0)],
            [], [(0, 1, 2, 3)])
        me.update()
        base = bpy.data.objects.new("board_base", me)
        base.data.materials.append(white)
        parent_collection.objects.link(base)

        quads = []
        for j in range(NY):
            for i in range(NX):
                if (i + j) % 2:
                    continue
                cx, cy = (i - (NX - 1) / 2) * SQ, (j - (NY - 1) / 2) * SQ
                me = bpy.data.meshes.new("sq")
                me.from_pydata(
                    [(cx - SQ / 2, cy - SQ / 2, 0.0003), (cx + SQ / 2, cy - SQ / 2, 0.0003),
                     (cx + SQ / 2, cy + SQ / 2, 0.0003), (cx - SQ / 2, cy + SQ / 2, 0.0003)],
                    [], [(0, 1, 2, 3)])
                me.update()
                ob = bpy.data.objects.new(f"sq_{i}_{j}", me)
                ob.data.materials.append(black)
                parent_collection.objects.link(ob)
                quads.append(ob)

        self.root = bpy.data.objects.new("board_root", None)
        parent_collection.objects.link(self.root)
        base.parent = self.root
        base.matrix_parent_inverse = Matrix.Identity(4)
        for ob in quads:
            ob.parent = self.root
            ob.matrix_parent_inverse = Matrix.Identity(4)

        # board->world with no tilt (used for the flat "table" render)
        self._flat = Matrix.Identity(4)
        self.set(Matrix.Identity(4), (0, 0, 0))

    def set(self, rot, loc):
        """rot: 3x3 board->world rotation (identity = flat, z up); loc: centre in world."""
        self.root.matrix_world = Matrix.Translation(Vector(loc)) @ rot.to_4x4()

    def corners_world(self):
        """4 outer corners of the full board (base plane) in world coords."""
        out = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(self.root.matrix_world @ Vector((sx * BASE_W / 2, sy * BASE_H / 2, 0)))
        return out


def world_to_cv(cam):
    """4x4 world -> OpenCV camera frame (metres): Blender camera frame with y and z flipped."""
    return Matrix.Diagonal((1, -1, -1, 1)) @ cam.matrix_world.inverted()


def build_scene():
    rng = random.Random(SEED)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            break
        except TypeError:
            continue
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 960

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.05, 0.05, 0.05)

    # Table: 2x2 m light-grey plane at z = 0
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    table = bpy.context.object
    table.data.materials.append(material("table", (0.75, 0.75, 0.75, 1), 0.6))

    # Lighting: key area light + weak fill; avoid blown-out highlights
    bpy.ops.object.light_add(type="AREA", location=(-0.25, -0.35, 0.75))
    key = bpy.context.object
    key.data.energy = 8
    key.data.size = 0.5
    key.rotation_euler = (math.radians(25), 0, 0)
    bpy.ops.object.light_add(type="AREA", location=(0.30, 0.30, 0.80))
    fill = bpy.context.object
    fill.data.energy = 2
    fill.data.size = 1.0

    # Stereo rig: fixed, no randomness. Both cameras identical, 150 mm baseline
    # along x, looking down (~25 deg off vertical) and slightly forward at origin.
    cams = []
    for name, side in (("left", -1), ("right", 1)):
        loc = Vector((side * BASELINE / 2, CAM_Y, CAM_H))
        bpy.ops.object.camera_add(location=loc)
        cam = bpy.context.object
        cam.name = name
        cam.data.lens = LENS_MM
        cam.data.sensor_width = SENSOR_MM
        cam.data.sensor_fit = "HORIZONTAL"
        cam.data.clip_start = 0.01
        cam.data.clip_end = 10
        direction = -(loc.normalized())  # aim at world origin
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cams.append(cam)
    scene.camera = cams[0]

    left_cam, right_cam = cams
    bpy.context.view_layer.update()
    M_left_m = world_to_cv(left_cam)
    M_right_m = world_to_cv(right_cam)

    board = Board(scene.collection)

    # Cylinders (hidden while calibrating): small touching clusters, fixed seed
    cyl_mats = {c: material(f"cyl_{c}", rgba, 0.4) for c, rgba in COLORS.items()}
    cyl_objs, cyl_data = [], []
    placed = []
    gap = 2 * CYL_R + 0.0005  # touching neighbours, 0.5 mm apart
    while len(placed) < 16:
        k = rng.choice([1, 1, 2, 3, 3])
        cx, cy = rng.uniform(-0.12, 0.12), rng.uniform(-0.09, 0.09)
        ang = rng.uniform(0, 2 * math.pi)
        # 1 = single, 2 = touching pair, 3 = touching triangle
        offs = [(0, 0), (gap, 0), (gap / 2, gap * math.sqrt(3) / 2)][:k]
        members = [(cx + ox * math.cos(ang) - oy * math.sin(ang), cy + ox * math.sin(ang) + oy * math.cos(ang))
                   for ox, oy in offs]
        if any(math.hypot(px - mx, py - my) < gap + 0.01 for px, py in placed for mx, my in members):
            continue
        placed.extend(members)
    for mx, my in placed[:16]:
        cyl_data.append({"color": rng.choice(list(COLORS)), "x": mx, "y": my})
    for i, c in enumerate(cyl_data):
        bpy.ops.mesh.primitive_cylinder_add(radius=CYL_R, depth=CYL_H,
                                            location=(c["x"], c["y"], CYL_H / 2), vertices=48)
        ob = bpy.context.object
        ob.name = f"cyl_{i}_{c['color']}"
        ob.data.materials.append(cyl_mats[c["color"]])
        cyl_objs.append(ob)
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))  # smooth side, sharp cap edges
        ob.hide_render = True  # shown only for the scene renders

    return scene, rng, left_cam, right_cam, M_left_m, M_right_m, board, cyl_data, cyl_objs


def visible_in_both(corners, scene, cams, lo=0.03, hi=0.97):
    for cam in cams:
        for c in corners:
            u, v, _ = world_to_camera_view(scene, cam, c)
            if not (lo <= u <= hi and lo <= v <= hi) or v <= 0:
                return False
    return True


def rand_board_pose(rng):
    """Random board pose over the workspace, height 0-250 mm, tilt <= 35 deg, any yaw."""
    yaw = rng.uniform(0, 2 * math.pi)
    tilt_axis_ang = rng.uniform(0, 2 * math.pi)
    tilt = math.radians(rng.uniform(5, 35))  # small tilt avoids grazing-light glare
    R = (Matrix.Rotation(tilt_axis_ang, 3, "Z")
         @ Matrix.Rotation(tilt, 3, "X")
         @ Matrix.Rotation(yaw, 3, "Z"))
    loc = Vector((rng.uniform(-0.16, 0.16), rng.uniform(-0.12, 0.12),
                  rng.uniform(0.02, 0.250)))
    return R, loc


def render(scene, path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"wrote {path}")


def main():
    os.makedirs(os.path.join(OUT, "calib"), exist_ok=True)
    scene, rng, left_cam, right_cam, M_left_m, M_right_m, board, cyl_data, cyl_objs = build_scene()

    w, h = scene.render.resolution_x, scene.render.resolution_y
    f_mm = LENS_MM / SENSOR_MM * w
    K = [[f_mm, 0, w / 2], [0, f_mm, h / 2], [0, 0, 1]]

    # ---- calibration poses -------------------------------------------------
    board_poses_mm = []
    for i in range(args.n_calib):
        for _ in range(500):
            R, loc = rand_board_pose(rng)
            board.set(R, loc)
            bpy.context.view_layer.update()
            corners = board.corners_world()
            if min(c.z for c in corners) > 0.01 and visible_in_both(corners, scene, (left_cam, right_cam)):
                break
        else:
            raise RuntimeError("could not place board in shared FOV")
        # board->left-cv-cam, mm
        M = (M_left_m @ board.root.matrix_world).copy()
        M_mm = M.copy()
        M_mm.translation *= 1000.0
        board_poses_mm.append([list(row) for row in M_mm])
        render(scene, os.path.join(OUT, "calib", f"left_{i:02d}.png"))
        scene.camera = right_cam
        render(scene, os.path.join(OUT, "calib", f"right_{i:02d}.png"))
        scene.camera = left_cam

    # ---- flat board on the table -------------------------------------------
    yaw = math.radians(6.0)
    board.set(Matrix.Rotation(yaw, 3, "Z"), (0.01, -0.02, BOARD_THICK))
    for _ in range(200):
        if visible_in_both(board.corners_world(), scene, (left_cam, right_cam)):
            break
        board.set(Matrix.Rotation(yaw, 3, "Z"), (rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), BOARD_THICK))
    T_note = ("table frame: origin = flat-board centre projected to the table plane z=0; "
              "x/y = the flat board's own axes (yaw only); z up. "
              "Note: this need not coincide with the origin/order OpenCV's "
              "findChessboardCorners would pick on the table_*.png images.")
    flat_world = board.root.matrix_world.copy()
    table_frame_R = flat_world.to_3x3()
    table_frame_t = Vector((flat_world.translation.x, flat_world.translation.y, 0.0))

    render(scene, os.path.join(OUT, "table_left.png"))
    scene.camera = right_cam
    render(scene, os.path.join(OUT, "table_right.png"))
    scene.camera = left_cam

    # ---- cylinder scene ------------------------------------------------------
    board.root.hide_render = True
    for ob in board.root.children:
        ob.hide_render = True
    for ob in cyl_objs:
        ob.hide_render = False
    # confirm every cylinder top is inside both views
    for c in cyl_data:
        top = Vector((c["x"], c["y"], CYL_H))
        for cam in (left_cam, right_cam):
            u, v, _ = world_to_camera_view(scene, cam, top)
            if not (0.02 <= u <= 0.98 and 0.02 <= v <= 0.98):
                raise RuntimeError(f"cylinder {c} outside {cam.name} view")

    cyl_json = []
    for c in cyl_data:
        top = Vector((c["x"], c["y"], CYL_H))
        px = {}
        for name, cam in (("left", left_cam), ("right", right_cam)):
            u, v, _ = world_to_camera_view(scene, cam, top)
            px[name] = [round(u * w, 3), round((1.0 - v) * h, 3)]  # v from TOP
        cyl_json.append({
            "color": c["color"],
            "center_table_world_mm": [round(c["x"] * 1000, 3), round(c["y"] * 1000, 3),
                                      round(CYL_H * 1000, 3)],
            "pixel_left": px["left"],
            "pixel_right": px["right"],
        })
    render(scene, os.path.join(OUT, "scene_left.png"))
    scene.camera = right_cam
    render(scene, os.path.join(OUT, "scene_right.png"))
    scene.camera = left_cam

    # ---- ground truth --------------------------------------------------------
    R_l, t_l = M_left_m.to_3x3(), M_left_m.translation * 1000.0
    R_r, t_r = M_right_m.to_3x3(), M_right_m.translation * 1000.0
    R_st = R_r @ R_l.transposed()
    T_st = t_r - R_st @ t_l

    M_w2cvL = M_left_m.copy()
    M_w2cvL.translation *= 1000.0

    # table-frame points -> left cv-cam frame, mm
    M_table_world = table_frame_R.to_4x4()
    M_table_world.translation = table_frame_t
    M_t2cvL = M_w2cvL @ M_table_world

    gt = {
        "image_size": [w, h],
        "K": K,
        "K_note": "OpenCV convention, fx = fy = lens/sensor_width*w; same for both cameras",
        "R": [list(row) for row in R_st],
        "T": [round(v, 6) for v in T_st],
        "T_note": "X_right = R @ X_left + T, mm; cv frames (x right, y down, z forward)",
        "world_to_left_cv": [list(row) for row in M_w2cvL],
        "world_to_left_cv_note": "4x4 world(mm) -> left OpenCV camera frame(mm)",
        "T_left_table": [list(row) for row in M_t2cvL],
        "table_frame_note": T_note,
        "board_poses": board_poses_mm,
        "board_poses_note": "4x4 board(mm, origin = board centre) -> left cv camera, one per calib pair",
        "square_mm": args.square_mm,
        "n_calib": args.n_calib,
        "cylinders": cyl_json,
    }
    import json
    with open(os.path.join(OUT, "ground_truth.json"), "w") as fp:
        json.dump(gt, fp, indent=1)
    print(f"wrote {os.path.join(OUT, 'ground_truth.json')}")


main()
