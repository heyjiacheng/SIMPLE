# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""The capped bottle of the uncap-bottle tasks: the PartNet bottle ``articulated:1`` with a cap that pulls off.

The stock asset (``assets/articulated/001``) models the cap on a slide joint along the bottle axis
(``articulate_joint_2``) and a hinge about it (``articulate_joint_0``), and leaves cap-bottle collisions off. This
module writes a variant of its MJCF next to the stock one and loads that instead. The variant differs in five ways:

* It is smaller, in one of two sizes (`SIZES`, the asset ids). The stock bottle is 10.6 cm wide and its cap 9.8 cm,
  about as far as a Dex3 hand opens.

  - ``bottle_cap:0`` (the uncap-bottle task's) is slimmer, so one Dex3 hand can hold the body and the other can
    grip the cap. It is 7.4 cm wide (cap 6.9 cm) and keeps the stock height: a 15.3 cm body and a 2.1 cm cap.
  - ``bottle_cap:1`` (the uncap-small-bottle task's) is 0.7 times that in every direction: 5.2 cm wide (cap
    4.8 cm), with a 10.7 cm body and a 1.5 cm cap. The hands that work on it hide more of it from the robot's head
    camera.
* The cap sits flush on the bottle and stays there when pressed. The stock slide range lets it sink 9 mm into the
  body, and the stock limit is soft.
* Dry friction holds the cap on harder than the bottle weighs. A hand that pulls only the cap lifts the whole
  capped bottle, so uncapping needs the other hand on the bottle. The bottle can also be tilted, shaken or turned
  upside down without losing its cap.
* The cap comes off. Once it has slid `CAP_CLEAR` along the bottle axis, it is a free body: it stays in the hand
  that holds it, falls when let go, and collides with the bottle like anything else. See `CapMechanism`.
* The masses are set: a 120 g bottle and a 20 g cap at size 0, 75 g and 8 g at size 1. The stock meshes give 39 g
  and 9 g, with a helper body of 1 g and 1e-4 kg m^2.

How the cap comes off: MuJoCo can't add or remove joints at run time, so the cap carries free-body joints from the
start. After the slide come three slides along the bottle's axes, then two hinges and the spin hinge about the cap's
centre. While the cap is on, equality constraints hold all six at zero, and the cap only slides on the friction joint.
`CapMechanism.update` turns the constraints off once the cap has slid far enough. The engines need no special case:
MuJoCo simulates the joints, and Isaac Sim replays them like any other ``articulate_joint*``.

For Isaac Sim, which replays MuJoCo's state, it also writes a matching USD (``output_usd/001_bottle_cap.usda``, and
``001_bottle_cap_small.usda`` for size 1). This happens whenever ``pxr`` can be imported, i.e. in Isaac Sim
processes. The stock USD is unscaled, references a layer on another machine, and lacks the ``/root/bottle`` layout
the engine expects.
"""

from __future__ import annotations

import importlib.util
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np

from simple.utils import resolve_data_path

from .articulate import Articulated_Object_Names, ArticulatedObjectAsset
from .asset_manager import AssetManager

_STOCK_ID = 1  # assets/articulated/001, "bottle"
_STOCK_MJCF = f"assets/articulated/{_STOCK_ID:03d}/output_mjcf/{_STOCK_ID:03d}.xml"
_STOCK_USD = f"assets/articulated/{_STOCK_ID:03d}/output_usd/{_STOCK_ID:03d}.usd"

_STOCK_SCALE = np.array([0.1, 0.1, 0.1])  # the stock MJCF's mesh scale


@dataclass(frozen=True)
class BottleSize:
    """One size of the variant."""

    # The meshes' scale. Mesh axes: y is the bottle axis (link_1's frame turns it to world z), x and z are radial.
    # Scale x and z alike, so the cap still turns about its own centre.
    scale: tuple[float, float, float]
    bottle_mass: float  # kg
    cap_mass: float  # kg
    # N: dry friction on the cap's slide joint, over 2.5 times the weight below it (the bottle). A hand pulling only
    # the cap lifts the capped bottle instead of uncapping it, and the cap stays on under shaking. Holding the bottle
    # and pulling the cap takes this plus the cap's weight, well within a Dex3 grip (several N per finger).
    hold_force: float
    # Armature on the cap's three hinges (kg m^2, under 2% of its inertia). Three hinges in a row lock up when the
    # middle one is at 90 degrees (gimbal lock), and without armature MuJoCo's mass matrix is singular there: a
    # tumbling cap blew up in about 1 in 4 throws.
    hinge_armature: float
    # The variant is written as <file>.xml next to the stock MJCF, so the mesh paths stay relative, and as
    # <file>.usda next to the stock USD.
    file: str


# The sizes by asset id: "bottle_cap:0" and "bottle_cap:1".
SIZES = {
    # 7.4 cm wide and 17.4 cm tall (cap 6.9 cm wide, 2.1 cm tall). The uncap-bottle task's size.
    "0": BottleSize(scale=(0.07, 0.1, 0.07), bottle_mass=0.12, cap_mass=0.02, hold_force=3.0, hinge_armature=1e-7,
                    file="001_bottle_cap"),
    # 0.7 times size 0 in every direction: 5.2 cm wide and 12.2 cm tall (cap 4.8 cm wide, 1.5 cm tall). The
    # uncap-small-bottle task's size.
    "1": BottleSize(scale=(0.049, 0.07, 0.049), bottle_mass=0.075, cap_mass=0.008, hold_force=2.0,
                    hinge_armature=2e-8, file="001_bottle_cap_small"),
}

# m the cap slides against the hold force before it comes free. The cap's collision mesh sits flush on top of the
# bottle's (there is no neck inside it), so any gap clears the bottle.
CAP_CLEAR = 0.01
# m: the slide's end stop. It is past CAP_CLEAR, so a fast pull frees the cap before reaching it; the check runs once
# per control step (20 ms).
_SLIDE_END = 0.03
# Armature on the slide (kg). Once the cap carries its own slide along the bottle axis, the two slides differ only by
# the helper body's 1 g between them, and MuJoCo scales the friction's softness by that small mass: the cap crept
# off at 1 N. With this the slide acts on 21 g again, like the stock joint, and holds to the hold force.
_SLIDE_ARMATURE = 0.02
# MuJoCo collides meshes as convex hulls with one contact point, and the flat cap sank about 8 mm into a table or
# the bottle top on it. Small spheres on both faces, standing 0.5 mm proud of them, give it four points to rest on.
_REST_SPHERES = 4  # per face
_REST_SPHERE_RADIUS = 0.003  # m
# Stiffer than MuJoCo's default friction impedance, so the held cap doesn't creep. The engine's noslip pass already
# stops creep; this keeps it small without noslip too.
_SOLIMP_FRICTION = "0.99 0.999 0.001 0.5 2"
# The cap's seat (the slide's lower limit) and the constraints that hold the cap on. With the default softness, a
# press sinks the 21 g that moves on the slide about 1 mm per newton; this keeps it under a millimetre. 0.01 s is 2
# control substeps (physics_dt 0.005).
_SOLREF_LIMIT = "0.01 1"
_SOLIMP_LIMIT = "0.95 0.99 0.001 0.5 2"

BOTTLE_JOINTS = dict(cap_slide="articulate_joint_2", cap_spin="articulate_joint_0", free="articulate_floating_base")
BOTTLE_BODIES = dict(bottle="link_1", cap="link_0")
# The cap's free-body joints, in the order they act: slides along the bottle's x, y (its axis) and z, then turns
# about x and z, then the stock spin about the axis. The names keep the ``articulate_joint`` prefix, which is what
# the engines sync.
CAP_FREE_JOINTS = dict(x="articulate_joint_cap_x", y="articulate_joint_cap_y", z="articulate_joint_cap_z",
                       rx="articulate_joint_cap_rx", rz="articulate_joint_cap_rz")
_CAP_FREE_DOFS = (("x", "slide", "1 0 0"), ("y", "slide", "0 1 0"), ("z", "slide", "0 0 1"),
                  ("rx", "hinge", "1 0 0"), ("rz", "hinge", "0 0 1"))
_CAP_LOCKED = (*CAP_FREE_JOINTS.values(), BOTTLE_JOINTS["cap_spin"])  # held at 0 while the cap is on


def _lock_name(joint: str) -> str:
    """Name of the equality constraint that holds `joint` at 0 while the cap is on."""
    return f"{joint}_lock"


def _floats(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()])


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _cylinder_inertial(body: ET.Element, mass: float, lo: np.ndarray, hi: np.ndarray) -> None:
    """Give `body` the inertia of a solid cylinder of `mass` filling the box lo..hi (its frame; axis along y)."""
    radius = 0.25 * ((hi[0] - lo[0]) + (hi[2] - lo[2]))
    height = hi[1] - lo[1]
    radial = mass * (3 * radius**2 + height**2) / 12
    axial = mass * radius**2 / 2
    for old in body.findall("inertial"):
        body.remove(old)
    inertial = ET.Element("inertial", pos=_fmt(0.5 * (lo + hi)), mass=_fmt([mass]),
                          diaginertia=_fmt([radial, axial, radial]))
    body.insert(0, inertial)


def bottle_cap_mjcf(stock_path: str, size: BottleSize) -> str:
    """The variant's MJCF at `size` (see the module docstring), built from the stock MJCF at `stock_path`."""
    import trimesh

    root = ET.parse(stock_path).getroot()
    root.set("model", "bottle_cap")
    compiler = root.find("compiler")
    mesh_dir = os.path.join(os.path.dirname(stock_path), compiler.get("meshdir", ".") if compiler is not None else ".")

    scale = np.asarray(size.scale)
    meshes = {mesh.get("name"): mesh for mesh in root.iter("mesh")}
    for mesh in meshes.values():
        stock = _floats(mesh.get("scale", "1 1 1"))
        assert np.allclose(stock, _STOCK_SCALE), f"unexpected stock mesh scale {stock} in {stock_path}"
        mesh.set("scale", _fmt(scale))
    k = scale / _STOCK_SCALE

    bodies = {body.get("name"): body for body in root.iter("body")}
    bottle, cap, helper = bodies["link_1"], bodies["link_0"], bodies["link_0_helper"]
    # the cap sits at the slide joint's anchor (helper) offset back by its geoms: scale both like the meshes
    helper.set("pos", _fmt(_floats(helper.get("pos")) * k))
    for geom in cap.findall("geom"):
        geom.set("pos", _fmt(_floats(geom.get("pos", "0 0 0")) * k))

    def bounds(body: ET.Element) -> tuple[np.ndarray, np.ndarray]:
        """Box around `body`'s collision meshes, in its frame."""
        points = []
        for geom in body.findall("geom"):
            if geom.get("class") != "collision":
                continue
            mesh = trimesh.load(os.path.join(mesh_dir, meshes[geom.get("mesh")].get("file")), force="mesh")
            points.append(np.asarray(mesh.vertices) * scale + _floats(geom.get("pos", "0 0 0")))
        points = np.concatenate(points)
        return points.min(0), points.max(0)

    _cylinder_inertial(bottle, size.bottle_mass, *bounds(bottle))
    cap_lo, cap_hi = bounds(cap)
    _cylinder_inertial(cap, size.cap_mass, cap_lo, cap_hi)
    for inertial in helper.findall("inertial"):  # massless in effect, without the stock's large inertia
        inertial.set("mass", "0.001")
        inertial.set("diaginertia", "1e-07 1e-07 1e-07")

    # grip like the graspnet objects: torsional friction (condim 4) helps hold the bottle and cap in a Dex3 hand
    for body in (bottle, cap):
        for geom in body.findall("geom"):
            if geom.get("class") == "collision":
                geom.set("condim", "4")
                geom.set("friction", "1 0.05 0.005")

    joints = {joint.get("name"): joint for joint in root.iter("joint")}
    slide, spin = joints[BOTTLE_JOINTS["cap_slide"]], joints[BOTTLE_JOINTS["cap_spin"]]
    slide.set("range", _fmt([0.0, _SLIDE_END]))  # 0: cap on, flush with the bottle
    slide.set("solreflimit", _SOLREF_LIMIT)
    slide.set("solimplimit", _SOLIMP_LIMIT)
    slide.set("frictionloss", _fmt([size.hold_force]))
    slide.set("solimpfriction", _SOLIMP_FRICTION)
    slide.set("damping", "0.5")
    slide.set("armature", _fmt([_SLIDE_ARMATURE]))

    # The cap's free-body joints (see the module docstring). The turns are about the cap's centre, which lies on the
    # spin axis. They carry no friction or damping: once the constraints let go, the cap is a plain free body.
    centre = 0.5 * (cap_lo + cap_hi)  # the cap's centre of mass, in its frame
    assert spin in list(cap), "the stock spin hinge is on the cap body"
    for key in ("frictionloss", "solimpfriction", "damping"):
        spin.attrib.pop(key, None)
    spin.set("pos", _fmt(centre))
    spin.set("armature", _fmt([size.hinge_armature]))
    at = list(cap).index(spin)
    for i, (key, kind, axis) in enumerate(_CAP_FREE_DOFS):
        joint = ET.Element("joint", name=CAP_FREE_JOINTS[key], type=kind, axis=axis, limited="false")
        if kind == "hinge":
            joint.set("pos", _fmt(centre))
            joint.set("armature", _fmt([size.hinge_armature]))
        cap.insert(at + i, joint)  # before the spin: MuJoCo applies a body's joints in order

    # spheres to rest on, on both flat faces (the cap's axis is its frame's y)
    collision = next(geom for geom in cap.findall("geom") if geom.get("class") == "collision")
    half = 0.5 * (cap_hi[1] - cap_lo[1])
    ring = 0.25 * ((cap_hi[0] - cap_lo[0]) + (cap_hi[2] - cap_lo[2])) - _REST_SPHERE_RADIUS - 0.002
    for side, sign in (("bottom", -1), ("top", 1)):
        for k in range(_REST_SPHERES):
            angle = 2 * np.pi * (k + 0.5) / _REST_SPHERES
            pos = centre + [ring * np.cos(angle), sign * (half - _REST_SPHERE_RADIUS + 0.0005), ring * np.sin(angle)]
            cap.append(ET.Element("geom", name=f"link_0_rest_{side}_{k}", type="sphere", pos=_fmt(pos),
                                  size=_fmt([_REST_SPHERE_RADIUS]), **{key: collision.get(key) for key in
                                                                      ("class", "condim", "friction")}))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for name in _CAP_LOCKED:
        ET.SubElement(equality, "joint", name=_lock_name(name), joint1=name, polycoef="0 0 0 0 0",
                      solref=_SOLREF_LIMIT, solimp=_SOLIMP_LIMIT)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


class CapMechanism:
    """The cap of a compiled MuJoCo scene that holds the bottle: how far it is off, and taking it off.

    One per compiled model; the engines compile a new one at every reset, which puts the cap back on.
    """

    def __init__(self, model, data) -> None:
        self.model, self.data = model, data
        self.bottle = model.body(BOTTLE_BODIES["bottle"]).id
        self.cap = model.body(BOTTLE_BODIES["cap"]).id
        self._slide = int(model.jnt_qposadr[model.joint(BOTTLE_JOINTS["cap_slide"]).id])
        self._shift = [int(model.jnt_qposadr[model.joint(CAP_FREE_JOINTS[k]).id]) for k in ("x", "y", "z")]
        self._locks = np.array([model.equality(_lock_name(name)).id for name in _CAP_LOCKED])
        self._geoms = np.array([g for g in range(model.ngeom)
                                if model.geom_bodyid[g] == self.cap and model.geom_conaffinity[g]])

    @property
    def free(self) -> bool:
        """Whether the cap has come off the bottle."""
        return not self.data.eq_active[self._locks].any()

    @property
    def off(self) -> float:
        """How far the cap's centre is from where it sits on the bottle (m). The turns are about that centre."""
        q = self.data.qpos
        x, y, z = q[self._shift]
        return float(np.linalg.norm([x, q[self._slide] + y, z]))  # slide and y both run along the bottle axis

    def update(self) -> bool:
        """Free the cap once it has slid `CAP_CLEAR` off the bottle. Call once per control step; returns `free`."""
        if not self.free and self.data.qpos[self._slide] >= CAP_CLEAR:
            self.data.eq_active[self._locks] = 0
            # Collide with the bottle from now on: its geoms have contype 0 and conaffinity 1, like the cap's. The
            # cap is clear of the bottle by now, so this adds no contact at once. MuJoCo's broad phase also filters
            # whole bodies by the contype/conaffinity of their geoms, worked out when compiling: update both.
            self.model.geom_contype[self._geoms] |= 1
            self.model.body_contype[self.cap] |= 1
        return self.free


def _quat_z_to(axis: np.ndarray):
    """The rotation (a Gf.Quatf) that turns +z onto `axis`."""
    from pxr import Gf

    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c = float(a[2])
    if c > 1 - 1e-9:
        return Gf.Quatf(1.0)
    if c < -1 + 1e-9:
        return Gf.Quatf(0.0, 1.0, 0.0, 0.0)  # half a turn about x
    s = np.sqrt(2 * (1 + c))
    v = np.cross([0.0, 0.0, 1.0], a) / s
    return Gf.Quatf(float(0.5 * s), float(v[0]), float(v[1]), float(v[2]))


def bottle_cap_usda(mjcf_path: str, name: str) -> str:
    """The variant as an Isaac Sim USD (text), laid out like the importer's articulated assets.

    Layout: /root/<name>/{base, <links>, joints/<MuJoCo joint names>, root_joint}. PhysX joints link two bodies, so
    each MuJoCo joint gets a link of its own. A body's last joint moves the body itself (`link_0_helper`, `link_0`);
    the cap's earlier joints move helper links in between. The meshes, joint axes and anchors come from the compiled
    MJCF, so both engines draw the same geometry. Isaac only replays MuJoCo's state (base pose, joint positions), so
    the links carry visuals and placeholder masses, with no colliders.
    """
    import mujoco
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # default pose: base upright at its origin, cap on and not turned
    base = model.body("articulate_base").id
    bottle = model.body(BOTTLE_BODIES["bottle"]).id
    origin = data.xpos[base].copy()
    assert np.allclose(data.xquat[base], [1, 0, 0, 0])

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/root").GetPrim())
    top = f"/root/{name}"
    UsdGeom.Xform.Define(stage, top)

    def visual_geom(body: int) -> int | None:
        return next((g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                     and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0), None)

    def visual_mesh(body: str, path: str, shift: np.ndarray) -> None:
        """`body`'s visual geom as a USD mesh at `path`, in the frame `shift` from the base origin."""
        g = visual_geom(model.body(body).id)
        m = model.geom_dataid[g]
        points = (model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                  @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - origin - shift)
        faces = model.mesh_face[model.mesh_faceadr[m]:model.mesh_faceadr[m] + model.mesh_facenum[m]]
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces.reshape(-1)]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        material = UsdShade.Material.Define(stage, f"{top}/Looks/{body}")
        shader = UsdShade.Shader.Define(stage, f"{top}/Looks/{body}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        rgb = model.mat_rgba[model.geom_matid[g]][:3] if model.geom_matid[g] >= 0 else model.geom_rgba[g][:3]
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def rigid_body(key: str, pos: np.ndarray, mass: float, mesh_of: str | None = None) -> None:
        xform = UsdGeom.Xform.Define(stage, f"{top}/{key}")
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass_api.CreateMassAttr(mass)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5, 1e-5, 1e-5))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        if mesh_of is not None:
            UsdGeom.Xform.Define(stage, f"{top}/{key}/visuals")
            visual_mesh(mesh_of, f"{top}/{key}/visuals/mesh", pos)

    kinds = {mujoco.mjtJoint.mjJNT_SLIDE: (UsdPhysics.PrismaticJoint, "linear"),
             mujoco.mjtJoint.mjJNT_HINGE: (UsdPhysics.RevoluteJoint, "angular")}

    def joint(j: int, body0: str, pos0: np.ndarray, body1: str, pos1: np.ndarray) -> None:
        """MuJoCo joint `j` between links `body0` and `body1`, which sit at `pos0` and `pos1` (from the base)."""
        kind, drive = kinds[model.jnt_type[j]]
        anchor, rot = data.xanchor[j] - origin, _quat_z_to(data.xaxis[j])  # the joint frame's +z is MuJoCo's axis
        usd = kind.Define(stage, f"{top}/joints/{model.joint(j).name}")
        usd.CreateBody0Rel().SetTargets([f"{top}/{body0}"])
        usd.CreateBody1Rel().SetTargets([f"{top}/{body1}"])
        usd.CreateLocalPos0Attr(Gf.Vec3f(*map(float, anchor - pos0)))
        usd.CreateLocalRot0Attr(rot)
        usd.CreateLocalPos1Attr(Gf.Vec3f(*map(float, anchor - pos1)))
        usd.CreateLocalRot1Attr(rot)
        usd.CreateAxisAttr("Z")
        if model.jnt_limited[j]:
            lower, upper = model.jnt_range[j] if drive == "linear" else np.degrees(model.jnt_range[j])
            usd.CreateLowerLimitAttr(float(lower))
            usd.CreateUpperLimitAttr(float(upper))
        api = UsdPhysics.DriveAPI.Apply(usd.GetPrim(), drive)  # Isaac zeroes the gains: the joints just follow MuJoCo
        api.CreateStiffnessAttr(0.0)
        api.CreateDampingAttr(0.0)

    # The bottle (articulate_base and link_1, welded together) is the base link. Every body below it moves on joints.
    rigid_body("base", np.zeros(3), float(model.body_mass[bottle]), model.body(bottle).name)
    links = {base: ("base", np.zeros(3)), bottle: ("base", np.zeros(3))}
    for b in range(bottle + 1, model.nbody):  # bodies come parent first
        if model.body_rootid[b] != base:
            continue
        name_b = model.body(b).name
        parent, parent_pos = links[model.body_parentid[b]]
        jnts = range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
        assert len(jnts) > 0, f"{name_b} is welded to its parent; give it a link of its own if that changes"
        for k, j in enumerate(jnts):
            if k == len(jnts) - 1:  # the body itself
                key, pos = name_b, data.xpos[b] - origin
                rigid_body(key, pos, float(model.body_mass[b]), name_b if visual_geom(b) is not None else None)
            else:  # a helper link at the joint's anchor
                key, pos = f"{name_b}_{model.joint(j).name.removeprefix('articulate_joint_')}", data.xanchor[j] - origin
                rigid_body(key, pos, 0.001)
            joint(j, parent, parent_pos, key, pos)
            parent, parent_pos = key, pos
        links[b] = (parent, parent_pos)

    root_joint = UsdPhysics.FixedJoint.Define(stage, f"{top}/root_joint")
    root_joint.CreateBody1Rel().SetTargets([f"{top}/base"])
    UsdPhysics.ArticulationRootAPI.Apply(root_joint.GetPrim())
    # typed like the importer writes them: Isaac sets enabledSelfCollisions when it loads the articulation
    for attr, kind, value in (("enabledSelfCollisions", Sdf.ValueTypeNames.Bool, False),
                              ("solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32),
                              ("solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 1)):
        root_joint.GetPrim().CreateAttribute(f"physxArticulation:{attr}", kind).Set(value)
    return stage.GetRootLayer().ExportToString()


def _write_if_changed(path: str, text: str) -> None:
    try:
        with open(path) as f:
            if f.read() == text:
                return
    except FileNotFoundError:
        pass
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)  # atomic, so parallel loaders never read a partial file


def _bottom_z(mjcf_path: str) -> float:
    """Height of the bottle's lowest point above the frame it is attached at (negative: below it)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottle = model.body(BOTTLE_BODIES["bottle"]).id
    lowest = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bottle or model.geom_conaffinity[g] == 0 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = model.mesh_vert[start:start + count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        lowest = min(lowest, float(world[:, 2].min()))
    return lowest


class BottleCapAsset(ArticulatedObjectAsset):
    """The capped bottle. `bottom_z`: height of its bottom relative to the frame it is placed at."""

    def __init__(self, *args, bottom_z: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bottom_z = bottom_z

    def to_dict(self) -> dict[str, Any]:
        state = super().to_dict()
        state["res_id"] = "bottle_cap"  # reload through this manager, not the stock "articulated" one
        return state


@AssetManager.register("bottle_cap")
class BottleCapAssetManager(AssetManager):
    """The capped bottle, one asset per size: "bottle_cap:0" and "bottle_cap:1" (see `SIZES`)."""

    def __init__(self) -> None:
        self._bottom_z: dict[str, float] = {}
        self._usd_written: set[str] = set()

    def load(self, asset_id: str = "0", articulate_init_joint_qpos: dict[str, float] | None = None) -> BottleCapAsset:
        asset_id = str(asset_id)
        assert asset_id in SIZES, f"bottle_cap has the assets {list(SIZES)}; got {asset_id!r}"
        size = SIZES[asset_id]
        name = Articulated_Object_Names[_STOCK_ID]
        stock = resolve_data_path(_STOCK_MJCF, auto_download=True)
        mjcf_path = os.path.join(os.path.dirname(stock), f"{size.file}.xml")
        _write_if_changed(mjcf_path, bottle_cap_mjcf(stock, size))
        if mjcf_path not in self._bottom_z:
            self._bottom_z[mjcf_path] = _bottom_z(mjcf_path)
        usd_path = os.path.join(os.path.dirname(resolve_data_path(_STOCK_USD, auto_download=True)), f"{size.file}.usda")
        if usd_path not in self._usd_written and importlib.util.find_spec("pxr") is not None:  # in Isaac Sim
            _write_if_changed(usd_path, bottle_cap_usda(mjcf_path, name))
            self._usd_written.add(usd_path)
        return BottleCapAsset(
            uid=asset_id,
            label=name,
            name=name,
            usd_path=usd_path,
            mjcf_path=mjcf_path,
            description="bottle with a cap that pulls straight off",
            articulate_init_joint_qpos=articulate_init_joint_qpos,
            bottom_z=self._bottom_z[mjcf_path],
        )

    def sample(self, exclude: list[str] | None = None) -> BottleCapAsset:
        return self.load(random.choice([i for i in SIZES if i not in (exclude or [])]))

    def __len__(self) -> int:
        return len(SIZES)
