from PIL import Image, ImageDraw, ImageFont
import math
import os
import numpy as np


W, H = 1600, 920
BG = (247, 249, 252)
TEXT = (28, 28, 34)
SUBTEXT = (80, 84, 92)
GRID = (225, 230, 238)
ROBOT = (35, 35, 42)
ROBOT2 = (96, 98, 106)
ROBOT_FILL = (244, 246, 250)
SHADOW = (200, 205, 214)

WORLD_X = (232, 76, 61)
WORLD_Y = (54, 124, 214)
WORLD_Z = (48, 160, 110)
BODY_X = (170, 78, 235)
BODY_Y = (40, 175, 132)
BODY_Z = (38, 140, 210)
HEAD_X = (170, 78, 235)
HEAD_Y = (40, 175, 132)
HEAD_Z = (38, 140, 210)


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


F_TITLE = font(34, bold=True)
F_HEAD = font(26, bold=True)
F_BODY = font(22)
F_SMALL = font(18)
F_CODE = font(20)
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


def make_camera(azimuth_deg=52.0, elevation_deg=18.0):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
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


CAM_RIGHT, CAM_UP, CAM_FWD = make_camera()


def project(p, center, scale=225.0):
    p = np.asarray(p, dtype=float)
    cam_x = float(np.dot(p, CAM_RIGHT))
    cam_y = float(np.dot(p, CAM_UP))
    cam_z = float(np.dot(p, CAM_FWD))
    d = 4.5
    f = d / (d - cam_z)
    x = center[0] + cam_x * scale * f
    y = center[1] - cam_y * scale * f
    return (x, y), cam_z


def draw_arrow(draw, p0, p1, color, width=5, head=15):
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
    pa, za = project(a, center=(0, 0))
    pb, zb = project(b, center=(0, 0))
    w = max(1, int(width * (0.7 + 0.3 * (1.0 - min(max((za + zb) / 8.0, -1.0), 1.0)))))
    draw.line([pa, pb], fill=color, width=w)


def axes(draw, origin, R, center, colors, prefix, scale=0.7, width=5):
    o2d, _ = project(origin, center=center)
    labels = [f"{prefix}x", f"{prefix}y", f"{prefix}z"]
    for axis, color, lab in zip([R[:, 0], R[:, 1], R[:, 2]], colors, labels):
        end = origin + axis * scale
        e2d, _ = project(end, center=center)
        draw_arrow(draw, o2d, e2d, color, width=width, head=14)
        draw.text((e2d[0] + 6, e2d[1] - 13), lab, fill=color, font=F_SMALL)


def grid(draw, center, size=(6.2, 4.6), step=0.5):
    xs = np.arange(-size[0], size[0] + 1e-6, step)
    ys = np.arange(-size[1], size[1] + 1e-6, step)
    for x in xs:
        pts = [np.array([x, y, 0.0]) for y in np.linspace(-size[1], size[1], 80)]
        draw_lines(draw, pts, GRID, center, width=1)
    for y in ys:
        pts = [np.array([x, y, 0.0]) for x in np.linspace(-size[0], size[0], 80)]
        draw_lines(draw, pts, GRID, center, width=1)


def draw_lines(draw, pts, color, center, width=1):
    p2 = [project(p, center=center)[0] for p in pts]
    draw.line(p2, fill=color, width=width)


def stickman_joints(root, R):
    # Same pose in both panels; just the coordinate frame changes.
    pelvis = root
    spine = root + R @ np.array([0.02, 0.00, 0.42])
    neck = root + R @ np.array([0.03, 0.00, 0.78])
    head = root + R @ np.array([0.05, 0.00, 1.10])
    shoulder = root + R @ np.array([0.00, 0.00, 0.70])

    hip_l = root + R @ np.array([0.00, 0.12, 0.00])
    hip_r = root + R @ np.array([0.00, -0.12, 0.00])
    knee_l = root + R @ np.array([0.20, 0.12, -0.48])
    knee_r = root + R @ np.array([0.16, -0.12, -0.42])
    foot_l = root + R @ np.array([0.45, 0.12, -0.95])
    foot_r = root + R @ np.array([0.40, -0.12, -0.88])

    elbow_l = root + R @ np.array([0.20, 0.32, 0.58])
    hand_l = root + R @ np.array([0.46, 0.52, 0.52])
    elbow_r = root + R @ np.array([0.18, -0.32, 0.52])
    hand_r = root + R @ np.array([0.42, -0.56, 0.36])

    return {
        "pelvis": pelvis, "spine": spine, "neck": neck, "head": head, "shoulder": shoulder,
        "hip_l": hip_l, "hip_r": hip_r, "knee_l": knee_l, "knee_r": knee_r,
        "foot_l": foot_l, "foot_r": foot_r, "elbow_l": elbow_l, "elbow_r": elbow_r,
        "hand_l": hand_l, "hand_r": hand_r,
    }


def draw_stickman(draw, joints, center):
    # Shadow
    c = joints["pelvis"]
    ring = [c + np.array([0.24 * math.cos(a), 0.24 * math.sin(a), 0.0]) for a in np.linspace(0, 2 * math.pi, 40)]
    draw_lines(draw, ring + [ring[0]], SHADOW, center, width=6)

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
        pa, _ = project(joints[a], center=center)
        pb, _ = project(joints[b], center=center)
        draw.line([pa, pb], fill=ROBOT, width=6)

    for k, p in joints.items():
        pp, _ = project(p, center=center)
        r = 6 if k != "head" else 9
        draw.ellipse((pp[0] - r, pp[1] - r, pp[0] + r, pp[1] + r), fill=ROBOT2 if k != "head" else ROBOT, outline=ROBOT)


def panel(draw, x0, title, subtitle, frame_label, use_heading=False):
    w = 690
    h = 700
    x1 = x0 + w
    y0 = 150
    y1 = y0 + h
    draw.rounded_rectangle((x0, y0, x1, y1), radius=28, fill=(255, 255, 255), outline=(217, 222, 231), width=2)
    draw.text((x0 + 28, y0 + 20), title, fill=TEXT, font=F_HEAD)
    draw.text((x0 + 28, y0 + 58), subtitle, fill=SUBTEXT, font=F_BODY)

    center = (x0 + 315, y0 + 355)
    grid(draw, center, size=(2.9, 2.0), step=0.5)

    # Ground plane border to give 3D depth.
    corners = [
        np.array([-2.8, -1.9, 0.0]),
        np.array([2.9, -1.9, 0.0]),
        np.array([2.9, 2.0, 0.0]),
        np.array([-2.8, 2.0, 0.0]),
    ]
    border = [project(p, center=center)[0] for p in corners] + [project(corners[0], center=center)[0]]
    draw.line(border, fill=(205, 212, 222), width=2)

    # Fixed world triad in a corner.
    origin = np.array([-2.1, -1.35, 0.0])
    axes(draw, origin, np.eye(3), center, [WORLD_X, WORLD_Y, WORLD_Z], "W-", scale=0.55, width=5)

    # One and the same pose.
    root = np.array([0.55, 0.10, 0.0])
    yaw = math.radians(42)
    pitch = math.radians(-15)
    roll = math.radians(18)
    R_body = rotz(yaw) @ roty(pitch) @ rotx(roll)
    R_heading = rotz(yaw)
    R = R_heading if use_heading else R_body
    joints = stickman_joints(root, R_body)
    draw_stickman(draw, joints, center)

    # Frame axes attached to pelvis.
    if use_heading:
        axes(draw, joints["pelvis"], R_heading, center, [HEAD_X, HEAD_Y, HEAD_Z], "H-", scale=0.78, width=6)
        label = "只对齐 yaw, z 轴保持竖直"
        formula = "H = Rz(yaw)^{-1} · World"
    else:
        axes(draw, joints["pelvis"], R_body, center, [BODY_X, BODY_Y, BODY_Z], "B-", scale=0.78, width=6)
        label = "跟随完整姿态, z 轴也会一起歪"
        formula = "B = R_world_to_body^{-1}"

    # The same pose annotation.
    draw.rounded_rectangle((x0 + 28, y1 - 150, x1 - 28, y1 - 28), radius=18, fill=(250, 251, 253), outline=(226, 231, 238), width=2)
    draw.text((x0 + 50, y1 - 132), frame_label, fill=TEXT, font=F_BOLD)
    draw.text((x0 + 50, y1 - 95), label, fill=SUBTEXT, font=F_BODY)
    draw.text((x0 + 50, y1 - 60), formula, fill=SUBTEXT, font=F_CODE)


def main():
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)

    draw.text((42, 28), "同一姿态下：本体坐标系 vs heading 坐标系", fill=TEXT, font=F_TITLE)
    draw.text((42, 72), "左边是完整本体坐标系，右边是只去掉 yaw 的 heading 坐标系。机器人姿态完全相同，只换坐标系定义。", fill=TEXT, font=F_BODY)

    panel(draw, 35, "本体坐标系 / Body Frame", "坐标轴跟随完整姿态一起转", "看这一边：z 轴会随身体倾斜", use_heading=False)
    panel(draw, 875, "heading 坐标系 / Heading Frame", "只跟随地面平面内的朝向", "看这一边：z 轴始终竖直", use_heading=True)

    out = os.path.join(os.getcwd(), "frame_compare_body_vs_heading.png")
    im.save(out)
    print(out)


if __name__ == "__main__":
    main()
