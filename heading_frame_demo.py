from PIL import Image, ImageDraw, ImageFont
import math
import os


W, H = 1000, 700
BG = (248, 249, 252)
WORLD_X = (230, 80, 70)
WORLD_Y = (60, 130, 210)
HEADING_X = (40, 170, 110)
HEADING_Y = (160, 110, 220)
ROBOT = (35, 35, 40)
ROBOT_FILL = (240, 243, 248)
TEXT = (30, 30, 35)
GRID = (224, 228, 236)


def load_font(size, bold=False):
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyhbd.ttc" if bold else r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT = load_font(24)
FONT_SMALL = load_font(18)
FONT_BOLD = load_font(26, bold=True)


def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return c, s


def world_to_px(x, y):
    # World frame is top-view: x right, y up
    cx, cy = 360, 360
    scale = 120
    return cx + x * scale, cy - y * scale


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


def draw_axes(draw, origin, theta, x_color, y_color, label_prefix, scale=85, width=5):
    ox, oy = origin
    c, s = rot(theta)
    x_end = (ox + scale * c, oy - scale * s)
    y_end = (ox - scale * s, oy - scale * c)
    draw_arrow(draw, origin, x_end, x_color, width=width)
    draw_arrow(draw, origin, y_end, y_color, width=width)
    draw.text((x_end[0] + 8, x_end[1] - 18), f"{label_prefix}x", fill=x_color, font=FONT_SMALL)
    draw.text((y_end[0] + 8, y_end[1] - 18), f"{label_prefix}y", fill=y_color, font=FONT_SMALL)


def draw_robot(draw, pos, yaw, body_len=86, body_w=54):
    px, py = pos
    c, s = rot(yaw)

    # Body rectangle corners in local frame
    corners = [
        (body_len / 2, body_w / 2),
        (body_len / 2, -body_w / 2),
        (-body_len / 2, -body_w / 2),
        (-body_len / 2, body_w / 2),
    ]
    pts = []
    for lx, ly in corners:
        wx = px + lx * c - ly * s
        wy = py - (lx * s + ly * c)
        pts.append((wx, wy))

    draw.polygon(pts, fill=ROBOT_FILL, outline=ROBOT)
    draw.line([pts[2], pts[0]], fill=ROBOT, width=3)

    # Heading arrow at the front
    front = (px + (body_len / 2 + 28) * c, py - (body_len / 2 + 28) * s)
    draw_arrow(draw, (px, py), front, ROBOT, width=6, head=14)
    draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=ROBOT)

    # Shoulder points for a rough stick figure
    shoulder = (px + 0.15 * body_len * c, py - 0.15 * body_len * s)
    left_hand = (
        shoulder[0] + 40 * math.cos(yaw + 1.1),
        shoulder[1] - 40 * math.sin(yaw + 1.1),
    )
    right_hand = (
        shoulder[0] + 40 * math.cos(yaw - 1.1),
        shoulder[1] - 40 * math.sin(yaw - 1.1),
    )
    draw.line([shoulder, left_hand], fill=(90, 90, 95), width=5)
    draw.line([shoulder, right_hand], fill=(90, 90, 95), width=5)
    draw.ellipse((left_hand[0] - 5, left_hand[1] - 5, left_hand[0] + 5, left_hand[1] + 5), fill=(90, 90, 95))
    draw.ellipse((right_hand[0] - 5, right_hand[1] - 5, right_hand[0] + 5, right_hand[1] + 5), fill=(90, 90, 95))


def make_frame(t, total):
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)

    # Background grid
    for gx in range(40, W, 80):
        draw.line((gx, 30, gx, H - 30), fill=GRID, width=1)
    for gy in range(30, H, 80):
        draw.line((40, gy, W - 40, gy), fill=GRID, width=1)

    # Title
    draw.text((40, 24), "Heading 坐标系动图", fill=TEXT, font=FONT_BOLD)
    draw.text((40, 62), "世界坐标系固定不动；heading 坐标系只跟随机器人 yaw 旋转", fill=TEXT, font=FONT)

    # World axes panel
    origin = (130, 560)
    draw_axes(draw, origin, 0.0, WORLD_X, WORLD_Y, "W-", scale=90, width=5)
    draw.text((62, 610), "世界坐标系", fill=TEXT, font=FONT_BOLD)
    draw.text((62, 642), "x 固定向右, y 固定向上", fill=TEXT, font=FONT_SMALL)

    # Motion path in world frame
    pts = []
    for i in range(total):
        u = i / (total - 1)
        x = -1.2 + 2.4 * u
        y = 0.15 * math.sin(2.2 * math.pi * u) + 0.2 * math.sin(0.8 * math.pi * u)
        pts.append(world_to_px(x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(180, 190, 205), width=4)
    for p in pts[::5]:
        draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(180, 190, 205))

    # Robot motion state
    u = t / (total - 1)
    x = -1.2 + 2.4 * u
    y = 0.15 * math.sin(2.2 * math.pi * u) + 0.2 * math.sin(0.8 * math.pi * u)
    yaw = -0.9 + 1.8 * u + 0.45 * math.sin(1.6 * math.pi * u)
    px, py = world_to_px(x, y)

    # Velocity arrow
    du = 1e-3
    u2 = min(1.0, u + du)
    x2 = -1.2 + 2.4 * u2
    y2 = 0.15 * math.sin(2.2 * math.pi * u2) + 0.2 * math.sin(0.8 * math.pi * u2)
    vx, vy = x2 - x, y2 - y
    norm = math.hypot(vx, vy) or 1.0
    vx, vy = vx / norm, vy / norm
    draw_arrow(draw, (px, py), (px + 75 * vx, py - 75 * vy), (120, 140, 160), width=4, head=12)
    draw.text((px + 80 * vx + 8, py - 80 * vy - 10), "v_world", fill=(120, 140, 160), font=FONT_SMALL)

    # Robot
    draw_robot(draw, (px, py), yaw)

    # Heading frame attached to robot
    draw_axes(draw, (px, py), yaw, HEADING_X, HEADING_Y, "H-", scale=70, width=5)
    draw.text((60, 100), "机器人在转身时，W 系不动，H 系跟着前向方向转", fill=TEXT, font=FONT_SMALL)

    # Explanatory box
    box = (650, 140, 955, 500)
    draw.rounded_rectangle(box, radius=22, fill=(255, 255, 255), outline=(216, 220, 228), width=2)
    draw.text((680, 170), "这一帧在说明什么", fill=TEXT, font=FONT_BOLD)
    lines = [
        "1. 机器人位置在世界里移动",
        "2. 机器人 yaw 改变时，heading 坐标系一起转",
        "3. heading 系只消掉朝向差异",
        "4. 前进/侧移在 H 系里更稳定",
    ]
    y0 = 225
    for line in lines:
        draw.text((680, y0), line, fill=TEXT, font=FONT)
        y0 += 52

    # Current state labels
    draw.text((680, 450), f"t = {t+1}/{total}", fill=(95, 95, 100), font=FONT_BOLD)
    draw.text((680, 485), f"yaw = {math.degrees(yaw):.1f}°", fill=(95, 95, 100), font=FONT_BOLD)

    return im


def main():
    frames = []
    total = 36
    for t in range(total):
        frames.append(make_frame(t, total))

    out = os.path.join(os.getcwd(), "heading_frame_demo.gif")
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    print(out)


if __name__ == "__main__":
    main()
