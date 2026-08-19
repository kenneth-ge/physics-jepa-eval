"""Reusable object -> MuJoCo geom builders.

MuJoCo convexifies meshes for collision, so concave shapes (the true bowl)
are assembled from convex box panels; convex shapes (the triangular prism)
can be plain meshes.
"""

import math

import numpy as np

from .rotations import rz, ry, quat_str
from .xml_scene import fr, DEFAULT_FRICTION


def bowl_geoms(o, idx):
    """A TRUE bowl standing on z=0: vertical outer wall ring, wide flat lip,
    parabolic interior — all convex box panels + a center disc."""
    R_out = float(o.get("outer_radius", 0.46))
    H = float(o.get("height", 0.10))          # lip height above ground
    lip = float(o.get("lip_width", 0.10))
    D = float(o.get("depth", min(H - 0.02, 0.30)))  # lip top -> interior bottom
    nseg = int(o.get("segments", 28))
    rings = int(o.get("rings", 6))
    th = float(o.get("thickness", 0.012))
    fric = o.get("friction", DEFAULT_FRICTION)
    rgba = o.get("rgba", [0.55, 0.55, 0.62, 1])
    pos = o.get("pos", [0, 0])
    px, py = pos[0], pos[1]
    pz = pos[2] if len(pos) > 2 else 0.0
    R_in = R_out - lip
    z_bot = H - D
    if z_bot < 0.005:
        raise ValueError(f"bowl {idx}: depth {D} leaves interior below ground (height {H})")
    if R_in <= 0.05:
        raise ValueError(f"bowl {idx}: lip_width {lip} leaves no interior (outer_radius {R_out})")

    geoms = []
    common = f'material="obj" rgba="{fr(rgba)}" condim="6" friction="{fr(fric)}"'
    tan_half = math.tan(math.pi / nseg)

    def add(name, size, pos, quat):
        geoms.append(f'<geom name="{name}" type="box" size="{fr(size)}" '
                     f'pos="{pos[0]+px:.5f} {pos[1]+py:.5f} {pos[2]+pz:.5f}" '
                     f'quat="{quat}" {common}/>')

    for i in range(nseg):
        t = 2 * math.pi * (i + 0.5) / nseg
        ct, st = math.cos(t), math.sin(t)
        q_flat = quat_str(rz(t))
        # outer wall: outer face exactly at R_out, from ground to lip top
        r_c = R_out - th
        add(f"bowl{idx}_wall{i}", [th, r_c * tan_half + 0.004, H / 2],
            [r_c * ct, r_c * st, H / 2], q_flat)
        # lip: top face at z=H, spanning R_in..R_out
        r_m = (R_in + R_out) / 2
        add(f"bowl{idx}_lip{i}", [lip / 2 + 0.004, r_m * tan_half + 0.004, th],
            [r_m * ct, r_m * st, H - th], q_flat)
        # interior: parabola z(r) = z_bot + D*(r/R_in)^2, from R_in down to r0
        r_edges = np.linspace(R_in, R_in / rings * 0.7, rings + 1)
        for k in range(rings):
            r1, r2 = r_edges[k], r_edges[k + 1]
            z1 = z_bot + D * (r1 / R_in) ** 2
            z2 = z_bot + D * (r2 / R_in) ** 2
            dr, dz = r2 - r1, z2 - z1
            L = math.hypot(dr, dz)
            alpha = math.atan2(dz, dr)
            n_r, n_z = -dz / L, dr / L
            if n_z < 0:
                n_r, n_z = -n_r, -n_z
            r_mid, z_mid = (r1 + r2) / 2, (z1 + z2) / 2
            cr = r_mid - n_r * th
            cz = z_mid - n_z * th
            add(f"bowl{idx}_in{i}_{k}",
                [L / 2 + 0.004, max(r_mid * tan_half, 0.012) + 0.004, th],
                [cr * ct, cr * st, cz], quat_str(rz(t) @ ry(-alpha)))
    r0 = float(r_edges[-1])
    geoms.append(f'<geom name="bowl{idx}_cap" type="cylinder" '
                 f'size="{r0 + 0.01:.4f} {th}" pos="{px} {py} {z_bot - th + 0.002 + pz:.5f}" '
                 f'{common}/>')
    return geoms


def ramp_geoms(o, idx):
    L = float(o.get("length", 0.6))
    W = float(o.get("width", 0.5))
    ang = math.radians(float(o.get("angle_deg", 12)))
    yaw = math.radians(float(o.get("yaw_deg", 0)))
    th = float(o.get("thickness", 0.012))
    fric = o.get("friction", DEFAULT_FRICTION)
    rgba = o.get("rgba", [0.6, 0.5, 0.4, 1])
    pos = o.get("pos", [0, 0])
    px, py = pos[0], pos[1]
    pz = pos[2] if len(pos) > 2 else 0.0
    # Sink the ramp so the TOP surface meets the ground exactly at the low
    # edge — otherwise the end face is a step that kills incoming balls.
    cz = pz + (L / 2) * math.sin(ang) - th * math.cos(ang)
    return [f'<geom name="ramp{idx}" type="box" size="{L/2} {W/2} {th}" '
            f'pos="{px} {py} {cz:.5f}" quat="{quat_str(rz(yaw) @ ry(-ang))}" '
            f'material="obj" rgba="{fr(rgba)}" condim="6" friction="{fr(fric)}"/>']


def prism_mesh_asset(name="prism", half_width=0.09, half_length=0.09, height=0.14):
    """Triangular prism mesh (convex, so MuJoCo collision is exact):
    triangle cross-section in x-z (base on the ground, apex up), length in y."""
    w, l, h = half_width, half_length, height
    verts = [(-w, -l, 0), (w, -l, 0), (0, -l, h),
             (-w, l, 0), (w, l, 0), (0, l, h)]
    flat = " ".join(f"{c:.4f}" for v in verts for c in v)
    return f'<mesh name="{name}" vertex="{flat}"/>'


def prism_geom(name, pos, yaw_deg=0.0, mesh="prism", rgba=(0.25, 0.7, 0.35, 1)):
    return (f'<geom name="{name}" type="mesh" mesh="{mesh}" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} 0" euler="0 0 {yaw_deg:.2f}" '
            f'material="obj" rgba="{fr(rgba)}"/>')


def pyramid_mesh_asset(name="pyramid", half=0.11, height=0.20):
    """Square-base pyramid mesh (convex): base on the ground, apex up."""
    h = half
    verts = [(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0), (0, 0, height)]
    flat = " ".join(f"{c:.4f}" for v in verts for c in v)
    return f'<mesh name="{name}" vertex="{flat}"/>'


def pyramid_geom(name, pos, yaw_deg=0.0, mesh="pyramid", rgba=(0.95, 0.55, 0.15, 1)):
    return (f'<geom name="{name}" type="mesh" mesh="{mesh}" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} 0" euler="0 0 {yaw_deg:.2f}" '
            f'material="obj" rgba="{fr(rgba)}"/>')
