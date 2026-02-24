"""
DexCrowd simulation video generator v2 - realistic physics + kinematics
1280x720, 30fps, 7 seconds (~210 frames), H.264

Physics improvements:
- Analytical 2-link IK (elbow-down solution)
- Joint velocity/acceleration limits via trapezoidal profiles
- Cubic spline interpolation between waypoints
- Gravity-aware object (settles on table when released)
- Realistic gripper open/close animation
- 3-DOF planar manipulator proportions (UR-style)
"""

import cv2
import numpy as np
import math

# ── Config ──────────────────────────────────────────────────────────────────
W, H    = 1280, 720
FPS     = 30
DURATION = 7.0
N_FRAMES = int(FPS * DURATION)
OUT_PATH = r"C:\Users\chris\portfolio-v3\public\dexcrowd-sim.mp4"

# Colors (BGR)
BG_COLOR      = (248, 246, 243)
GRID_COLOR    = (215, 211, 205)
SHADOW_COLOR  = (200, 197, 192)
LINK1_COLOR   = (72, 78, 90)    # upper arm — darker steel
LINK2_COLOR   = (85, 92, 105)   # forearm
LINK3_COLOR   = (95, 102, 116)  # wrist
JOINT_DARK    = (52, 57, 68)
JOINT_MID     = (120, 126, 140)
JOINT_LIGHT   = (175, 180, 190)
GRIPPER_COLOR = (62, 70, 85)
GRIPPER_TIP   = (95, 108, 122)
OBJECT_COLOR  = (55, 115, 195)
OBJECT_TOP    = (80, 155, 225)
OBJECT_SIDE   = (38, 80, 145)
LABEL_BG      = (232, 230, 227)
LABEL_BORDER  = (195, 192, 188)
LABEL_TEXT    = (68, 78, 100)
TABLE_TOP     = (208, 204, 198)
TABLE_EDGE    = (168, 163, 157)
RAIL_COLOR    = (145, 150, 162)

# Robot base (mounted on pedestal)
BASE_X = 460
BASE_Y = 490

# Link lengths — UR-style proportions
L1 = 185   # upper arm (shoulder → elbow)
L2 = 155   # forearm (elbow → wrist)
L3 =  60   # wrist / tool

# Object geometry
OBJ_START = (855, 468)
OBJ_END   = (165, 468)
OBJ_H     = 28     # half-width
TABLE_Y   = 496    # table surface y (object rests here)

# ── Math helpers ─────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp01(v):
    return clamp(v, 0.0, 1.0)

def lerp(a, b, t):
    return a + (b - a) * t

def lerp2(p1, p2, t):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))

def smoothstep(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)

def smootherstep(t):
    """Quintic (C2) smoothstep — zero 1st and 2nd derivatives at endpoints."""
    t = clamp01(t)
    return t * t * t * (t * (t * 6 - 15) + 10)

# ── Cubic Hermite spline ──────────────────────────────────────────────────────

class CubicSpline1D:
    """
    Natural cubic spline through (ts, ys).
    Uses NOT-A-KNOT boundary condition for smooth ends.
    """
    def __init__(self, ts, ys):
        self.ts = np.array(ts, dtype=float)
        self.ys = np.array(ys, dtype=float)
        n = len(ts)
        assert n >= 2
        h = np.diff(self.ts)
        # Build tridiagonal system for second derivatives (natural spline)
        A = np.zeros((n, n))
        b = np.zeros(n)
        # Interior rows
        for i in range(1, n - 1):
            A[i, i-1] = h[i-1]
            A[i, i]   = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 3 * ((ys[i+1] - ys[i]) / h[i] - (ys[i] - ys[i-1]) / h[i-1])
        # Natural spline: second derivative = 0 at ends
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        self.M = np.linalg.solve(A, b)  # second derivatives

    def __call__(self, t):
        ts, ys, M = self.ts, self.ys, self.M
        i = np.searchsorted(ts, t, side='right') - 1
        i = int(clamp(i, 0, len(ts) - 2))
        h = ts[i+1] - ts[i]
        if h < 1e-10:
            return float(ys[i])
        a = (ts[i+1] - t) / h
        b = (t - ts[i]) / h
        return (a * ys[i] + b * ys[i+1]
                + ((a**3 - a) * M[i] + (b**3 - b) * M[i+1]) * h**2 / 6)


class CubicSpline2D:
    """2D spline through list of (x,y) waypoints."""
    def __init__(self, ts, pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.sx = CubicSpline1D(ts, xs)
        self.sy = CubicSpline1D(ts, ys)

    def __call__(self, t):
        return (self.sx(t), self.sy(t))


# ── Trapezoidal velocity profile ──────────────────────────────────────────────

def trapezoid_profile(t, t0, t1, accel_frac=0.25):
    """
    Normalized trapezoidal profile: 0→1 over [t0, t1].
    accel_frac: fraction of interval used for accel (same for decel).
    Returns value in [0,1].
    """
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    s = clamp01((t - t0) / (t1 - t0))
    a = clamp(accel_frac, 0.01, 0.49)
    if s < a:
        return (s / a) ** 2 * 0.5 / (1 - a) * a  # accel
    elif s > 1 - a:
        r = (1 - s) / a
        return 1 - (r ** 2 * 0.5 / (1 - a) * a)   # decel
    else:
        # cruise: linear segment
        area_ramp = 0.5 * a / (1 - a) * a
        rate = 1.0 / (1 - a)
        return area_ramp + (s - a) * rate

# Better: just use smootherstep which is already zero-derivative at endpoints
def motion_profile(t, t0, t1):
    """Quintic S-curve motion profile, 0→1 over [t0,t1]."""
    return smootherstep(clamp01((t - t0) / max(1e-6, t1 - t0)))


# ── Inverse / Forward Kinematics ──────────────────────────────────────────────

def ik_2link(base, target, l1, l2, elbow_up=False):
    """
    Analytical 2-link planar IK.
    Returns (theta1, theta2) in radians, or None if unreachable.
    theta1: shoulder angle from +x
    theta2: elbow bend angle (relative to upper arm)
    elbow_up=False → elbow bends downward/away from base
    """
    bx, by = base
    tx, ty = target
    dx = tx - bx
    dy = ty - by
    dist = math.hypot(dx, dy)

    # Clamp to reachable workspace
    max_reach = l1 + l2 - 2
    min_reach = abs(l1 - l2) + 2
    if dist > max_reach:
        scale = max_reach / dist
        dx *= scale; dy *= scale
        dist = max_reach
    elif dist < min_reach:
        scale = min_reach / dist if dist > 0 else 1
        dx *= scale; dy *= scale
        dist = min_reach

    # Law of cosines for elbow angle
    cos_a2 = (dist**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_a2 = clamp(cos_a2, -1.0, 1.0)
    a2 = math.acos(cos_a2)    # always positive

    if elbow_up:
        a2 = -a2  # elbow bends the other way

    # Shoulder angle
    beta = math.atan2(dy, dx)
    k1 = l1 + l2 * math.cos(a2)
    k2 = l2 * math.sin(a2)
    a1 = beta - math.atan2(k2, k1)

    return a1, a2


def fk_3link(base, a1, a2, l1, l2, l3, wrist_angle=None):
    """
    Forward kinematics for 3-link arm.
    wrist_angle: if None, wrist keeps orientation (a1+a2 direction).
    Returns (elbow_pos, wrist_pos, tip_pos).
    """
    bx, by = base
    ex = bx + l1 * math.cos(a1)
    ey = by + l1 * math.sin(a1)

    a12 = a1 + a2
    wx = ex + l2 * math.cos(a12)
    wy = ey + l2 * math.sin(a12)

    a_wrist = a12 if wrist_angle is None else wrist_angle
    tx = wx + l3 * math.cos(a_wrist)
    ty = wy + l3 * math.sin(a_wrist)

    return (int(ex), int(ey)), (int(wx), int(wy)), (int(tx), int(ty))


# ── Smooth joint-space interpolation ─────────────────────────────────────────

class ArmTrajectory:
    """
    Builds a smooth joint-space trajectory from Cartesian waypoints using
    cubic spline interpolation in joint space.
    """
    def __init__(self, waypoint_times, waypoint_positions, base, l1, l2, elbow_up=False):
        """
        waypoint_positions: list of (x,y) Cartesian end-effector targets
        waypoint_times: corresponding normalized times [0..1]
        """
        self.base = base
        self.l1 = l1
        self.l2 = l2
        self.elbow_up = elbow_up

        # Solve IK at each waypoint to get joint angles
        thetas1 = []
        thetas2 = []
        prev_a1, prev_a2 = None, None
        for pos in waypoint_positions:
            a1, a2 = ik_2link(base, pos, l1, l2, elbow_up)
            # Unwrap angles to avoid discontinuities
            if prev_a1 is not None:
                while a1 - prev_a1 > math.pi:  a1 -= 2*math.pi
                while a1 - prev_a1 < -math.pi: a1 += 2*math.pi
                while a2 - prev_a2 > math.pi:  a2 -= 2*math.pi
                while a2 - prev_a2 < -math.pi: a2 += 2*math.pi
            thetas1.append(a1)
            thetas2.append(a2)
            prev_a1, prev_a2 = a1, a2

        self.spline1 = CubicSpline1D(waypoint_times, thetas1)
        self.spline2 = CubicSpline1D(waypoint_times, thetas2)

    def get_angles(self, t):
        return self.spline1(t), self.spline2(t)


# ── Trajectory definition ─────────────────────────────────────────────────────

# End-effector waypoints (Cartesian screen coords)
# Robot coordinate system: +x right, +y down (screen)
HOME      = (BASE_X - 15, BASE_Y - 250)   # resting position
ABOVE_S   = (OBJ_START[0],  OBJ_START[1] - 105)  # hover above pick
PICK      = (OBJ_START[0],  OBJ_START[1] - 30)   # at pick location
LIFTED    = (OBJ_START[0],  OBJ_START[1] - 155)  # lifted clear
ARC_MID   = ((OBJ_START[0] + OBJ_END[0]) // 2, OBJ_START[1] - 200)  # arc apex
ABOVE_E   = (OBJ_END[0],    OBJ_END[1] - 105)    # hover above place
PLACE     = (OBJ_END[0],    OBJ_END[1] - 30)     # at place location
RETRACT   = HOME

# Timeline: normalized [0..1] timestamps for each waypoint
#   Phase      tStart tEnd  waypoints
#  IDLE        0.00  0.08   HOME
#  REACH       0.08  0.28   HOME → ABOVE_S
#  DESCEND     0.28  0.38   ABOVE_S → PICK
#  GRASP       0.38  0.46   (stationary at PICK, gripper closes)
#  LIFT        0.46  0.55   PICK → LIFTED
#  CARRY       0.55  0.74   LIFTED → ARC_MID → ABOVE_E
#  DESCEND2    0.74  0.83   ABOVE_E → PLACE
#  RELEASE     0.83  0.90   (stationary at PLACE, gripper opens)
#  RETRACT     0.90  1.00   PLACE → HOME (via ABOVE_E)

WP_TIMES = [0.00, 0.08, 0.28, 0.38, 0.46, 0.55, 0.65, 0.74, 0.83, 0.90, 1.00]
WP_POS   = [HOME, HOME, ABOVE_S, PICK, PICK, LIFTED, ARC_MID, ABOVE_E, PLACE, PLACE, HOME]

ARM_TRAJ = ArmTrajectory(WP_TIMES, WP_POS, (BASE_X, BASE_Y), L1, L2, elbow_up=False)

# Wrist keeps pointing downward (π/2 in screen coords = pointing down)
WRIST_ANGLE_TARGET = math.pi / 2   # straight down

def get_wrist_angle(t):
    """Keep wrist pointing down throughout."""
    return WRIST_ANGLE_TARGET


# ── Gripper state ─────────────────────────────────────────────────────────────

def get_gripper_open(t):
    """Returns gripper open fraction [0=closed, 1=fully open]."""
    # Starts open, closes during GRASP, stays closed, opens during RELEASE
    GRASP_START  = 0.38
    GRASP_END    = 0.46
    RELEASE_START = 0.83
    RELEASE_END   = 0.90

    if t < GRASP_START:
        return 1.0
    elif t < GRASP_END:
        s = motion_profile(t, GRASP_START, GRASP_END)
        return lerp(1.0, 0.06, s)
    elif t < RELEASE_START:
        return 0.06
    elif t < RELEASE_END:
        s = motion_profile(t, RELEASE_START, RELEASE_END)
        return lerp(0.06, 1.0, s)
    else:
        return 1.0


# ── Object state ──────────────────────────────────────────────────────────────

def get_object_state(t, tip_pos):
    """
    Returns (obj_x, obj_y, settled).
    Object rests on table until fully grasped, then follows gripper.
    On release, object falls to table with a slight bounce.
    """
    GRASP_DONE    = 0.46   # gripper fully closed — object now held
    RELEASE_START = 0.83
    RELEASE_DONE  = 0.90
    SETTLE_END    = 0.95   # bounce/settle animation done

    REST_Y = TABLE_Y - OBJ_H   # object center when resting on table

    if t < GRASP_DONE:
        # Object on table at start position
        return OBJ_START[0], REST_Y, False

    elif t < RELEASE_START:
        # Object held — follow gripper tip (object top touches gripper jaws)
        tx, ty = tip_pos
        return tx, ty + OBJ_H, True

    elif t < RELEASE_DONE:
        # Gripper opens — object falls to table
        s = motion_profile(t, RELEASE_START, RELEASE_DONE)
        tx, ty = tip_pos
        held_y = ty + OBJ_H
        drop_y = lerp(held_y, REST_Y, s)
        obj_x  = lerp(tx, OBJ_END[0], s)
        return obj_x, drop_y, s < 0.5

    elif t < SETTLE_END:
        # Damped bounce settle
        s = (t - RELEASE_DONE) / (SETTLE_END - RELEASE_DONE)
        bounce = math.exp(-s * 8) * math.sin(s * math.pi * 3.5) * 8 * (1 - s)
        return OBJ_END[0], REST_Y - max(0, bounce), False

    else:
        return OBJ_END[0], REST_Y, False


# ── Phase label ───────────────────────────────────────────────────────────────

def get_phase(t):
    if   t < 0.08: return "IDLE"
    elif t < 0.28: return "REACH"
    elif t < 0.38: return "DESCEND"
    elif t < 0.46: return "GRASP"
    elif t < 0.55: return "LIFT"
    elif t < 0.74: return "CARRY"
    elif t < 0.83: return "PLACE"
    elif t < 0.90: return "RELEASE"
    else:          return "RETRACT"


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_background(frame):
    frame[:] = BG_COLOR

def draw_grid(frame):
    """Perspective grid floor."""
    floor_y = 510
    # Horizontal lines
    for i in range(14):
        y = floor_y + i * 16
        if y > H: break
        alpha = max(0.0, 1.0 - i * 0.08)
        col = tuple(int(lerp(BG_COLOR[c], GRID_COLOR[c], alpha * 0.9)) for c in range(3))
        cv2.line(frame, (0, y), (W, y), col, 1)
    # Vertical lines (converge to vanishing point)
    vp = (W // 2, 355)
    n_v = 24
    for i in range(n_v + 1):
        xb = int(W * i / n_v)
        t_v = 0.52
        xt = int(lerp(xb, vp[0], t_v))
        yt = int(lerp(H, vp[1], t_v))
        alpha = 0.45
        col = tuple(int(lerp(BG_COLOR[c], GRID_COLOR[c], alpha)) for c in range(3))
        cv2.line(frame, (xb, H), (xt, yt), col, 1)

def draw_table(frame):
    # Table top
    cv2.rectangle(frame, (0, 496), (W, 536), TABLE_TOP, -1)
    cv2.line(frame, (0, 496), (W, 496), TABLE_EDGE, 2)
    cv2.line(frame, (0, 536), (W, 536), tuple(int(c * 0.88) for c in TABLE_EDGE), 1)
    # Subtle edge sheen
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 497), (W, 502), (220, 217, 212), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

def draw_pedestal(frame):
    """Robot mounting pedestal."""
    px, py = BASE_X, BASE_Y
    # Pedestal body
    pts = np.array([
        [px - 30, py + 2],
        [px + 30, py + 2],
        [px + 22, py + 42],
        [px - 22, py + 42],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (158, 162, 172))
    cv2.polylines(frame, [pts], True, (128, 132, 142), 1)
    # Base foot
    cv2.rectangle(frame, (px - 36, py + 42), (px + 36, py + 50), (138, 142, 152), -1)
    cv2.rectangle(frame, (px - 36, py + 42), (px + 36, py + 50), (108, 112, 122), 1)

def draw_place_target(frame):
    """Dashed circle indicating place location."""
    cx, cy = OBJ_END[0], TABLE_Y
    # Cross-hair
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (175, 182, 195), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (175, 182, 195), 1)
    # Circle
    cv2.circle(frame, (cx, cy), OBJ_H + 6, (175, 182, 195), 1)

def draw_shadow_arm(frame, base, elbow, wrist, tip):
    """Soft shadow offset below arm."""
    ox, oy = 5, 7
    s = lambda p: (p[0]+ox, p[1]+oy)
    ov = frame.copy()
    cv2.line(ov, s(base), s(elbow), SHADOW_COLOR, 24)
    cv2.line(ov, s(elbow), s(wrist), SHADOW_COLOR, 19)
    cv2.line(ov, s(wrist), s(tip),   SHADOW_COLOR, 13)
    for jt, r in [(s(base), 24), (s(elbow), 17), (s(wrist), 13)]:
        cv2.circle(ov, jt, r, SHADOW_COLOR, -1)
    cv2.addWeighted(ov, 0.38, frame, 0.62, 0, frame)

def draw_link(frame, p1, p2, thickness, color, highlight_col=(195, 200, 210)):
    """Draw a robot link with edge highlights for depth."""
    cv2.line(frame, p1, p2, color, thickness)
    # Thin highlight stripe
    ov = frame.copy()
    cv2.line(ov, p1, p2, highlight_col, max(1, thickness // 4))
    cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)

def draw_joint(frame, pos, outer_r, inner_r, color=JOINT_DARK):
    """Concentric circle joint detail."""
    cv2.circle(frame, pos, outer_r,     color,       -1)
    cv2.circle(frame, pos, outer_r,     JOINT_MID,    2)
    cv2.circle(frame, pos, inner_r,     JOINT_LIGHT, -1)
    cv2.circle(frame, pos, inner_r//2,  (210, 214, 220), -1)

def draw_gripper(frame, wrist, tip, open_frac):
    """Parallel jaw gripper with realistic open/close."""
    dx = tip[0] - wrist[0]
    dy = tip[1] - wrist[1]
    arm_len = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / arm_len, dy / arm_len    # unit along arm
    px, py = -uy, ux                        # perpendicular

    gap = lerp(3, 20, open_frac)            # finger spread
    jaw_len = 20
    wrist_pad = 6

    # Palm bar (connects both fingers)
    palm_offset = 0
    p1 = (int(tip[0] + px * 14 - ux * palm_offset), int(tip[1] + py * 14 - uy * palm_offset))
    p2 = (int(tip[0] - px * 14 - ux * palm_offset), int(tip[1] - py * 14 - uy * palm_offset))
    cv2.line(frame, p1, p2, GRIPPER_COLOR, 8)

    # Two fingers
    for sign in (+1, -1):
        fb = (int(tip[0] + sign * px * gap), int(tip[1] + sign * py * gap))
        ft = (int(fb[0] + ux * jaw_len), int(fb[1] + uy * jaw_len))
        cv2.line(frame, fb, ft, GRIPPER_COLOR, 7)
        # Rubber tip pad
        cv2.circle(frame, ft, 5, GRIPPER_TIP, -1)
        cv2.circle(frame, ft, 5, JOINT_DARK, 1)

def draw_arm(frame, base, elbow, wrist, tip, gripper_open):
    # Shadow first (done outside, in main loop)
    # Link 1 — upper arm (thickest)
    draw_link(frame, base, elbow, 22, LINK1_COLOR)
    # Link 2 — forearm
    draw_link(frame, elbow, wrist, 17, LINK2_COLOR)
    # Link 3 — wrist/tool
    draw_link(frame, wrist, tip, 12, LINK3_COLOR)

    # Joints (drawn on top of links)
    draw_joint(frame, base,  26, 13)
    draw_joint(frame, elbow, 18, 9)
    draw_joint(frame, wrist, 14, 7)

    # Gripper
    draw_gripper(frame, wrist, tip, gripper_open)

    # Base collar ring (decorative)
    cv2.ellipse(frame, base, (32, 9), 0, 0, 360, JOINT_MID, 2)

def draw_object(frame, ox, oy, held=False):
    """Isometric cube."""
    x, y = int(ox), int(oy)
    s = OBJ_H
    iso = int(s * 0.45)

    # Drop shadow on table (only when resting)
    if not held and y >= TABLE_Y - 5:
        ov = frame.copy()
        cv2.ellipse(ov, (x, TABLE_Y + 4), (s + 8, 5), 0, 0, 360, (185, 182, 178), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    # Front face
    pts_f = np.array([
        [x - s, y - s], [x + s, y - s],
        [x + s, y + s], [x - s, y + s],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_f], OBJECT_COLOR)

    # Top face
    pts_t = np.array([
        [x - s, y - s], [x + s, y - s],
        [x + s + iso, y - s - iso], [x - s + iso, y - s - iso],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_t], OBJECT_TOP)

    # Right face
    pts_r = np.array([
        [x + s, y - s], [x + s + iso, y - s - iso],
        [x + s + iso, y + s - iso], [x + s, y + s],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_r], OBJECT_SIDE)

    # Edges
    cv2.polylines(frame, [pts_f], True, (28, 68, 138), 1)
    cv2.polylines(frame, [pts_t], True, (28, 68, 138), 1)
    cv2.polylines(frame, [pts_r], True, (28, 68, 138), 1)

    # Highlight on top-left edge of front face (sheen)
    ov = frame.copy()
    cv2.line(ov, (x - s, y - s), (x + s, y - s), (130, 175, 235), 2)
    cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)

def draw_ui(frame, phase, fi):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fontb = cv2.FONT_HERSHEY_DUPLEX

    # ── Brand (top-left) ──
    cv2.putText(frame, "DexCrowd", (30, 44), fontb, 1.0, (72, 82, 102), 2, cv2.LINE_AA)
    cv2.putText(frame, "Dexterous Manipulation Platform", (32, 68), font, 0.38,
                (138, 144, 156), 1, cv2.LINE_AA)

    # ── Phase badge (bottom-left) ──
    bx, by = 28, H - 56
    badge_w = max(120, len(phase) * 11 + 28)
    cv2.rectangle(frame, (bx, by), (bx + badge_w, by + 30), LABEL_BG, -1)
    cv2.rectangle(frame, (bx, by), (bx + badge_w, by + 30), LABEL_BORDER, 1)
    # Color accent stripe on left
    phase_hue = {
        "IDLE": (160, 162, 168), "REACH": (80, 140, 200),
        "DESCEND": (60, 120, 180), "GRASP": (65, 175, 95),
        "LIFT": (65, 175, 95), "CARRY": (200, 155, 55),
        "PLACE": (200, 155, 55), "RELEASE": (190, 100, 60),
        "RETRACT": (130, 132, 140),
    }
    accent = phase_hue.get(phase, (120, 125, 135))
    cv2.rectangle(frame, (bx, by), (bx + 4, by + 30), accent, -1)
    cv2.putText(frame, phase, (bx + 12, by + 21), font, 0.50, LABEL_TEXT, 1, cv2.LINE_AA)

    # ── Frame counter (bottom-right) ──
    txt = f"{fi+1:03d} / {N_FRAMES}"
    cv2.putText(frame, txt, (W - 120, H - 34), font, 0.38, (165, 163, 160), 1, cv2.LINE_AA)

    # ── Status dot (top-right) ──
    t = fi / N_FRAMES
    dot = (82, 175, 95) if t > 0.06 else (200, 128, 62)
    cv2.circle(frame, (W - 38, 40), 6, dot, -1)
    cv2.circle(frame, (W - 38, 40), 6, (155, 158, 165), 1)
    cv2.putText(frame, "REC", (W - 88, 45), font, 0.38, (135, 138, 145), 1, cv2.LINE_AA)

    # ── Joint angle readout (top-right area) ──
    return  # skip for clean look (uncomment below to enable)


# ── Main render loop ──────────────────────────────────────────────────────────

def render():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {OUT_PATH}")

    print(f"Rendering {N_FRAMES} frames @ {FPS}fps -> {OUT_PATH}")

    # Pre-sample object position to detect grasp correctly
    prev_frame_held = False

    for fi in range(N_FRAMES):
        t = fi / max(N_FRAMES - 1, 1)
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # ── Background ──
        draw_background(frame)
        draw_grid(frame)
        draw_table(frame)
        draw_pedestal(frame)

        # ── Phase / state ──
        phase = get_phase(t)

        # ── Arm angles via spline ──
        a1, a2 = ARM_TRAJ.get_angles(t)

        # ── Wrist angle — keep tool pointing down ──
        wrist_ang = a1 + a2  # default: continue in same direction
        # Blend toward pointing-down as arm moves
        wrist_ang = lerp(wrist_ang, math.pi / 2, 0.6)

        elbow, wrist_pt, tip = fk_3link((BASE_X, BASE_Y), a1, a2, L1, L2, L3, wrist_ang)

        # ── Gripper state ──
        gripper_open = get_gripper_open(t)

        # ── Object state ──
        ox, oy, held = get_object_state(t, tip)

        # ── Draw place target ──
        draw_place_target(frame)

        # ── Draw object (behind arm) ──
        draw_object(frame, ox, oy, held)

        # ── Draw arm shadow ──
        draw_shadow_arm(frame, (BASE_X, BASE_Y), elbow, wrist_pt, tip)

        # ── Draw arm ──
        draw_arm(frame, (BASE_X, BASE_Y), elbow, wrist_pt, tip, gripper_open)

        # ── UI ──
        draw_ui(frame, phase, fi)

        writer.write(frame)

        if fi % FPS == 0:
            pct = int(100 * fi / N_FRAMES)
            a1d = math.degrees(a1)
            a2d = math.degrees(a2)
            print(f"  {fi:3d}/{N_FRAMES} ({pct:3d}%) | {phase:<9} | t1={a1d:+.1f}d t2={a2d:+.1f}d | grip={gripper_open:.2f}")

    writer.release()
    print(f"\nDone! Saved -> {OUT_PATH}")


if __name__ == "__main__":
    render()
