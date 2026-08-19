"""Scene XML assembly: one camera/lighting/floor convention for all evals.

Every eval family renders through the same fixed camera and lighting so
that embeddings are comparable across families.
"""

DEFAULT_CAMERA = dict(pos=[0, -2.8, 1.1], xyaxes=[1, 0, 0, 0, 0.366, 0.93], fovy=52)
DEFAULT_FRICTION = [1.0, 0.01, 0.002]


def fr(values):
    """Space-joined attribute string."""
    return " ".join(str(v) for v in values)


def scene_xml(*, geoms="", bodies="", assets="", floor=None, camera=None,
              extra_cameras_xml="", ground_friction=(1.0, 0.01, 0.02), timestep=0.002):
    """Standard scene skeleton. `floor=None` gives the default ground plane;
    pass an explicit geom string (e.g. an hfield) to replace it.
    `extra_cameras_xml` appends additional <camera> elements (see cameras.py)."""
    cam = {**DEFAULT_CAMERA, **(camera or {})}
    if floor is None:
        floor = (f'<geom name="floor" type="plane" size="4 4 0.1" material="ground" '
                 f'condim="6" friction="{fr(ground_friction)}"/>')
    return f"""
<mujoco model="scene">
  <option timestep="{timestep}"/>
  <visual>
    <global offwidth="1024" offheight="1024"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <material name="ground" rgba="0.72 0.72 0.76 1" specular="0.2" shininess="0.3"/>
    <material name="obj" specular="0.3" shininess="0.4"/>
    {assets}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <camera name="fixed" pos="{fr(cam['pos'])}" xyaxes="{fr(cam['xyaxes'])}" fovy="{cam['fovy']}"/>
    {extra_cameras_xml}
    {floor}
    {geoms}
    {bodies}
  </worldbody>
</mujoco>"""
