from PIL import Image, ImageDraw, ImageFont
import math
import os
import numpy as np


W, H = 1280, 860
BG = (247, 249, 252)
TEXT = (28, 28, 34)
GRID = (225, 230, 238)
WORLD_X = (231, 76, 60)
WORLD_Y = (54, 124, 214)
WORLD_Z = (48, 160, 110)
HEADING_X = (131, 92, 230)
HEADING_Y = (32, 170, 125)
HEADING_Z = (38, 140, 210)
ROBOT = (40, 40, 46)
ROBOT2 = (90, 92, 100)
ROBOT_FILL = (244, 246, 250)
SHADOW = (200, 205, 214)


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


F_TITLE = font(32, bold=True)
F_BODY = font(24)
F_SMALL = font(18)
F_BOLD = font(24, bold=True)


def rotz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def world_to_camera(azimuth_deg=45.0, elevation_deg=22.0):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    # Camera looks at the origin from a direction defined by azimuth/elevation.
    # Build a basis: right, up, forward.
    forward = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ])
    forward = forward / np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(up_world, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


CAM_RIGHT, CAM_UP, CAM_FWD = world_to_camera(azimuth_deg=52.0, elevation_deg=18.0)


def project(p, center=(545, 470), scale=220.0, perspective=True):
    p = np.asarray(p, dtype=float)
    cam_x = float(np.dot(p, CAM_RIGHT))
    cam_y = float(np.dot(p, CAM_UP))
    cam_z = float(np.dot(p, CAM_FWD))
    if perspective:
        d = 4.4
        f = d / (d - cam_z)
    else:
        f = 1.0
    x = center[0] + cam_x * scale * f
    y = center[1] - cam_y * scale * f
    return (x, y), cam_z


def draw_arrow(draw, p0, p1, color, width=5, head=16):
    draw.line([p0, p1], fill=color, width=width)
    x0, y0 = p0
    x1, y1 = p1
    ang = math.atan2(y1 - y0, x1 - x0)
    a1 = ang + math.radians(155)
    a2 = ang - math.radians(155)
    p2 = (x1 + head * math.cos(a1), y1 + head * math.sin(a1))
    p3 = (x1 + head * math.cos(a2), y1 + head * math.sin(a2))
    draw.polygon([p1, p2, p3], fill=color)


def draw_3d_segment(draw, a, b, color, width=5):
    pa, za = project(a)
    pb, zb = project(b)
    # Slight depth hint: segments farther away are a bit thinner.
    w = max(1, int(width * (0.72 + 0.28 * (1.0 - min(max((za + zb) / 8.0, -1.0), 1.0)))))
    draw.line([pa, pb], fill=color, width=w)


def draw_axes(draw, origin, R, colors, prefix, scale=0.55, width=5):
    o2d, _ = project(origin)
    axes = [R[:, 0], R[:, 1], R[:, 2]]
    labels = [f"{prefix}x", f"{prefix}y", f"{prefix}z"]
    for axis, col, lab in zip(axes, colors, labels):
        end3d = origin + axis * scale
        e2d, _ = project(end3d)
        draw_arrow(draw, o2d, e2d, col, width=width, head=14)
        draw.text((e2d[0] + 6, e2d[1] - 14), lab, fill=col, font=F_SMALL)


def polyline(draw, points, color, width=3, dash=False):
    pts = [project(p)[0] for p in points]
    if len(pts) < 2:
        return
    if not dash:
        draw.line(pts, fill=color, width=width)
        return
    for i in range(0, len(pts) - 1, 2):
        draw.line([pts[i], pts[i + 1]], fill=color, width=width)


def draw_grid(draw):
    # Floor grid on z=0 plane
    xs = np.linspace(-3.0, 3.0, 13)
    ys = np.linspace(-2.4, 2.4, 11)
    for x in xs:
        pts = [np.array([x, y, 0.0]) for y in np.linspace(-2.4, 2.4, 80)]
        polyline(draw, pts, GRID, width=1)
    for y in ys:
        pts = [np.array([x, y, 0.0]) for x in np.linspace(-3.0, 3.0, 80)]
        polyline(draw, pts, GRID, width=1)


def make_robot_pose(t, total):
    u = t / (total - 1)
    # Path and heading: move forward, then turn more strongly near the end.
    x = -1.55 + 3.1 * u
    y = 0.35 * math.sin(2.3 * math.pi * u) + 0.18 * math.sin(0.85 * math.pi * u)
    z = 0.0
    yaw = -0.95 + 2.2 * u + 0.55 * math.sin(1.7 * math.pi * u)
    pitch = 0.08 * math.sin(2.0 * math.pi * u)
    roll = 0.16 * math.sin(2.4 * math.pi * u)
    root = np.array([x, y, z])
    R = rotz(yaw) @ roty(pitch) @ rotx(roll)
    return root, R, yaw


def stickman_joints(root, R, t, total):
    u = t / (total - 1)
    # Body proportions in robot local frame.
    pelvis = root
    spine = root + R @ np.array([0.0, 0.0, 0.42])
    neck = root + R @ np.array([0.0, 0.0, 0.78])
    head = root + R @ np.array([0.0, 0.0, 1.08])
    shoulder = root + R @ np.array([0.0, 0.0, 0.70])

    # Legs swing while walking.
    leg_swing = 0.55 * math.sin(2 * math.pi * (u * 2.0))
    knee_lift = 0.16 * max(0.0, math.sin(2 * math.pi * (u * 2.0)))
    arm_swing = -0.45 * math.sin(2 * math.pi * (u * 2.0))
    bend = 0.12 * math.sin(4 * math.pi * u)

    # Local limb endpoints in the robot frame.
    hip_l = np.array([0.0, 0.12, 0.0])
    hip_r = np.array([0.0, -0.12, 0.0])
    foot_l = np.array([0.42 + 0.10 * math.cos(2 * math.pi * u), 0.12, -0.95 + 0.30 * knee_lift])
    foot_r = np.array([0.42 + 0.10 * math.cos(2 * math.pi * u + math.pi), -0.12, -0.95 + 0.30 * (1.0 - knee_lift)])
    knee_l = np.array([0.18, 0.12, -0.45 + 0.12 * knee_lift])
    knee_r = np.array([0.18, -0.12, -0.45 + 0.12 * (1.0 - knee_lift)])

    elbow_l = np.array([0.18, 0.33, 0.52 + 0.08 * bend])
    hand_l = np.array([0.42, 0.56, 0.30 + 0.07 * math.sin(2 * math.pi * u)])
    elbow_r = np.array([0.18, -0.33, 0.52 - 0.08 * bend])
    hand_r = np.array([0.42, -0.56, 0.30 - 0.07 * math.sin(2 * math.pi * u)])

    # A little body lean when stepping.
    spine = root + R @ np.array([0.02 * math.sin(2 * math.pi * u), 0.0, 0.42])
    neck = root + R @ np.array([0.03 * math.sin(2 * math.pi * u), 0.0, 0.78])
    head = root + R @ np.array([0.03 * math.sin(2 * math.pi * u), 0.0, 1.08])

    def tr(v):
        return root + R @ v

    joints = {
        "pelvis": pelvis,
        "spine": spine,
        "neck": neck,
        "head": head,
        "shoulder": shoulder,
        "hip_l": tr(hip_l),
        "hip_r": tr(hip_r),
        "knee_l": tr(knee_l),
        "knee_r": tr(knee_r),
        "foot_l": tr(foot_l),
        "foot_r": tr(foot_r),
        "elbow_l": tr(elbow_l),
        "hand_l": tr(hand_l),
        "elbow_r": tr(elbow_r),
        "hand_r": tr(hand_r),
    }
    return joints


def draw_stickman(draw, joints):
    # Ground contact shadow
    center = joints["pelvis"]
    shadow_pts = [center + np.array([0.22 * math.cos(a), 0.22 * math.sin(a), 0.0]) for a in np.linspace(0, 2 * math.pi, 40)]
    polyline(draw, shadow_pts + [shadow_pts[0]], SHADOW, width=6)

    # Body segments
    segs = [
        ("pelvis", "spine"),
        ("spine", "neck"),
        ("neck", "head"),
        ("shoulder", "elbow_l"),
        ("elbow_l", "hand_l"),
        ("shoulder", "elbow_r"),
        ("elbow_r", "hand_r"),
        ("pelvis", "hip_l"),
        ("hip_l", "knee_l"),
        ("knee_l", "foot_l"),
        ("pelvis", "hip_r"),
        ("hip_r", "knee_r"),
        ("knee_r", "foot_r"),
        ("shoulder", "pelvis"),
    ]
    for a, b in segs:
        draw_3d_segment(draw, joints[a], joints[b], ROBOT, width=6)

    # Joints
    for k, p in joints.items():
        pp, _ = project(p)
        r = 6 if k != "head" else 9
        draw.ellipse((pp[0] - r, pp[1] - r, pp[0] + r, pp[1] + r), fill=ROBOT2 if k != "head" else ROBOT, outline=ROBOT)


def make_frame(t, total):
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)

    draw_grid(draw)

    # Title and explanatory text
    draw.text((38, 28), "3D 火柴人版 heading 坐标系", fill=TEXT, font=F_TITLE)
    draw.text((38, 74), "世界坐标系固定在地面；heading 坐标系只跟随机器人在平面内的朝向变化", fill=TEXT, font=F_BODY)

    root, R, yaw = make_robot_pose(t, total)
    joints = stickman_joints(root, R, t, total)

    # Trajectory trail
    trail = [make_robot_pose(i, total)[0] for i in range(t + 1)]
    trail3 = [p + np.array([0.0, 0.0, 0.01]) for p in trail]
    polyline(draw, trail3, (180, 190, 206), width=4)

    # World axes at scene origin
    scene_origin = np.array([-2.45, -1.95, 0.0])
    draw_axes(draw, scene_origin, np.eye(3), [WORLD_X, WORLD_Y, WORLD_Z], "W-", scale=0.62, width=5)
    draw.text((60, 780), "世界坐标系", fill=TEXT, font=F_BOLD)
    draw.text((60, 810), "固定不动", fill=TEXT, font=F_SMALL)

    # Robot
    draw_stickman(draw, joints)

    # Heading axes attached to pelvis
    heading_R = rotz(yaw)
    draw_axes(draw, joints["pelvis"], heading_R, [HEADING_X, HEADING_Y, HEADING_Z], "H-", scale=0.72, width=6)

    # Explanatory panel
    x0, y0, x1, y1 = 890, 160, 1230, 610
    draw.rounded_rectangle((x0, y0, x1, y1), radius=24, fill=(255, 255, 255), outline=(216, 221, 229), width=2)
    draw.text((920, 190), "这帧怎么看", fill=TEXT, font=F_BOLD)
    bullets = [
        "W 轴始终固定在地面",
        "机器人在走路、转身、摆臂",
        "H 轴跟着 pelvis 的 yaw 转",
        "H 不会把整个人的姿态都吸进去",
    ]
    yy = 245
    for b in bullets:
        draw.text((920, yy), b, fill=TEXT, font=F_BODY)
        yy += 63

    draw.text((920, 540), f"frame {t+1}/{total}", fill=(86, 90, 99), font=F_BOLD)
    draw.text((920, 572), f"yaw = {math.degrees(yaw):.1f}°", fill=(86, 90, 99), font=F_BOLD)

    return im


def main():
    total = 42
    frames = [make_frame(t, total) for t in range(total)]
    out = os.path.join(os.getcwd(), "heading_frame_3d_stickman.gif")
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=85,
        loop=0,
        optimize=False,
    )
    print(out)


if __name__ == "__main__":
    main()
