"""Render simple test scenes with red and blue marbles.

Usage:
    blender -b -P render_scenes.py -- [--n 5] [--out renders]
"""
import argparse
import math
import os
import random
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--out", default="renders")
args = ap.parse_args(argv)

COLORS = {"red": (0.8, 0.02, 0.02, 1), "blue": (0.02, 0.08, 0.8, 1)}
TABLE_COLORS = [(0.55, 0.55, 0.53, 1), (0.35, 0.25, 0.15, 1), (0.15, 0.15, 0.15, 1), (0.6, 0.65, 0.6, 1)]
BALL_R = 0.008  # 16 mm marble


def material(name, rgba, roughness, transmission=0.0):
    m = bpy.data.materials.new(name)
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Transmission Weight"].default_value = transmission
    return m


def build_scene(seed):
    rng = random.Random(seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
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

    # Table
    bpy.ops.mesh.primitive_plane_add(size=2)
    bpy.context.object.data.materials.append(material("table", rng.choice(TABLE_COLORS), 0.7))

    # Marbles, non-overlapping inside a ~20x15 cm workspace
    placed = []
    n_balls = rng.randint(6, 16)
    while len(placed) < n_balls:
        x, y = rng.uniform(-0.1, 0.1), rng.uniform(-0.075, 0.075)
        if all(math.hypot(x - px, y - py) > 2.05 * BALL_R for px, py in placed):
            placed.append((x, y))
    mats = {c: material(c, rgba, 0.15) for c, rgba in COLORS.items()}
    for i, (x, y) in enumerate(placed):
        color = rng.choice(list(COLORS))
        bpy.ops.mesh.primitive_uv_sphere_add(radius=BALL_R, location=(x, y, BALL_R), segments=48, ring_count=24)
        ball = bpy.context.object
        ball.name = f"{color}_{i}"
        bpy.ops.object.shade_smooth()
        ball.data.materials.append(mats[color])

    # Lights: a key area light + a softer fill
    bpy.ops.object.light_add(type="AREA", location=(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.6))
    key = bpy.context.object
    key.data.energy = rng.uniform(4, 12)
    key.data.size = 0.4
    key.rotation_euler = (0, 0, 0)
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 0.8))
    bpy.context.object.data.energy = 2
    bpy.context.object.data.size = 1.0

    # Camera roughly top-down with some tilt, aimed at the workspace centre
    tilt = math.radians(rng.uniform(0, 30))
    yaw = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(0.28, 0.38)
    cam_loc = (dist * math.sin(tilt) * math.cos(yaw), dist * math.sin(tilt) * math.sin(yaw), dist * math.cos(tilt))
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    target = bpy.data.objects.new("target", None)
    scene.collection.objects.link(target)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam.data.lens = 35
    scene.camera = cam
    return scene


for i in range(args.n):
    scene = build_scene(seed=i)
    scene.render.filepath = os.path.abspath(f"{args.out}/scene_{i:02d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"wrote {scene.render.filepath}")
