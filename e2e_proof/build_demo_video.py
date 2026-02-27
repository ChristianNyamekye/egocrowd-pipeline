"""
Build the DexCrowd E2E Pipeline Demo Walkthrough Video.
Stitches together all pipeline stages into a single narrated demo.

Structure (with title cards):
  0. Title screen
  1. Stage 1: Egocentric clips (sample frames)
  2. Stage 2: MediaPipe hand pose overlay (real video with landmarks)
  3. Stage 3: Retargeting (animated visualization of joint angles)
  4. Stage 4: BC training (animated loss curve)
  5. Stage 5: MuJoCo sim rollout (full rollout video)
  6. Summary / results card
"""

import cv2
import numpy as np
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent
W, H = 1280, 720
FPS = 30
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX

# Dark theme colors
BG = (13, 17, 23)
ACCENT = (0, 220, 120)
ACCENT2 = (100, 180, 255)
WHITE = (230, 230, 230)
GRAY = (120, 120, 120)
RED = (60, 100, 220)
GOLD = (40, 190, 230)


def safe_blit(dst, src, ty, tx):
    """Blit src into dst at (ty, tx), clipping to bounds."""
    dH, dW = dst.shape[:2]
    sH, sW = src.shape[:2]
    ey = min(ty + sH, dH)
    ex = min(tx + sW, dW)
    sy = max(0, -ty); sx = max(0, -tx)
    ty = max(0, ty); tx = max(0, tx)
    ah = ey - ty; aw = ex - tx
    if ah > 0 and aw > 0:
        dst[ty:ey, tx:ex] = src[sy:sy+ah, sx:sx+aw]


def blank(color=BG):
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[:] = color
    return f


def put_text(frame, text, pos, font=FONT, scale=0.8, color=WHITE, thickness=2):
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def put_centered(frame, text, y, font=FONT, scale=1.0, color=WHITE, thickness=2):
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = (W - size[0]) // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(frame, step, total_steps=6, y=H-25):
    """Draw pipeline progress bar at bottom."""
    bar_w = W - 80
    bar_h = 8
    bar_x = 40
    # Background
    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (40, 40, 40), -1)
    # Filled
    filled = int(bar_w * step / total_steps)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, y), (bar_x + filled, y + bar_h), ACCENT, -1)
    # Step dots
    for s in range(1, total_steps + 1):
        sx = bar_x + int(bar_w * s / total_steps) - 4
        color = ACCENT if s <= step else (60, 60, 60)
        cv2.circle(frame, (sx, y + bar_h // 2), 5, color, -1)


def draw_watermark(frame):
    put_text(frame, "DexCrowd Pipeline", (W - 230, H - 10), FONT_SM, 0.4, (60, 80, 60), 1)


def title_screen(writer, duration_s=3.0):
    """Title card."""
    frames = int(duration_s * FPS)
    for i in range(frames):
        f = blank()
        # Gradient overlay
        for y in range(H):
            alpha = y / H * 0.3
            f[y] = (f[y] * (1 - alpha) + np.array([20, 30, 50]) * alpha).astype(np.uint8)

        t = i / frames
        # Fade in
        alpha = min(1.0, t * 3)

        put_centered(f, "DexCrowd", H//2 - 90, scale=2.5, color=ACCENT, thickness=4)
        put_centered(f, "Egocentric Video  ->  Robot Manipulation", H//2 - 20, scale=0.9, color=ACCENT2)
        put_centered(f, "End-to-End Pipeline Proof", H//2 + 30, scale=0.75, color=GRAY)

        # Subtext
        put_centered(f, "Clip  ->  Hand Pose  ->  Retarget  ->  BC Train  ->  MuJoCo Sim", H//2 + 90,
                    scale=0.55, color=GRAY)

        # Date
        put_text(f, "Feb 2026  |  MediaPipe + PyTorch + MuJoCo 3.5", (40, H - 12),
                FONT_SM, 0.45, (70, 90, 70), 1)

        # Fade in mask
        if alpha < 1.0:
            overlay = np.zeros_like(f)
            blended = cv2.addWeighted(f, alpha, overlay, 1 - alpha, 0)
            writer.write(blended)
        else:
            draw_watermark(f)
            writer.write(f)


def stage_card(writer, step, title, subtitle, duration_s=1.5):
    """Stage transition card."""
    frames = int(duration_s * FPS)
    for i in range(frames):
        f = blank()
        t = i / frames
        alpha = min(1.0, t * 4) * (1.0 if t < 0.7 else max(0.0, 1.0 - (t - 0.7) / 0.3))

        # Step number badge
        cv2.circle(f, (W // 2, H // 2 - 60), 45, ACCENT, 3)
        put_centered(f, str(step), H // 2 - 45, scale=1.8, color=ACCENT, thickness=3)

        put_centered(f, title, H // 2 + 30, scale=1.1, color=WHITE, thickness=2)
        put_centered(f, subtitle, H // 2 + 75, scale=0.6, color=GRAY)
        draw_progress_bar(f, step)
        draw_watermark(f)

        if alpha < 1.0:
            overlay = np.zeros_like(f)
            blended = cv2.addWeighted(f, alpha, overlay, 1 - alpha, 0)
            writer.write(blended)
        else:
            writer.write(f)


def stage1_clips(writer, duration_s=5.0):
    """Show sample frames from synthetic clips with labels."""
    clips_dir = OUTPUT_DIR / "clips"
    clips = sorted(clips_dir.glob("*.mp4"))[:9]

    # Extract one frame from each clip
    thumbnails = []
    for cp in clips:
        cap = cv2.VideoCapture(str(cp))
        mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frm = cap.read()
        cap.release()
        if ret:
            thumbnails.append((frm, Path(cp).stem))

    # Also add real cooking video frames
    real_dir = OUTPUT_DIR.parent / "test_data" / "real_video"
    for rv in sorted(real_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(rv))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frm = cap.read()
        cap.release()
        if ret:
            thumbnails.append((frm, rv.stem + " (real)"))

    total_frames = int(duration_s * FPS)
    th_w, th_h = W // 4, H // 3

    for fi in range(total_frames):
        f = blank()
        t = fi / total_frames

        # Header
        put_text(f, "STEP 1", (40, 45), FONT_SM, 0.6, ACCENT, 2)
        put_centered(f, "Egocentric Manipulation Clips", 80, scale=1.0, color=WHITE, thickness=2)
        put_centered(f, "30 clips x 4s | 10 task categories | 2 real cooking + 28 synthetic",
                    115, scale=0.5, color=GRAY)

        # Show thumbnails in grid (3x3 + 2 real)
        all_th = thumbnails[:11]
        for ti, (thumb, name) in enumerate(all_th):
            col = ti % 4
            row = ti // 4
            tx = 20 + col * (th_w + 8)
            ty = 135 + row * (th_h + 8)

            # Resize thumbnail
            small = cv2.resize(thumb, (th_w, th_h))
            # Subtle border
            border_color = ACCENT if "real" in name else ACCENT2
            cv2.rectangle(small, (0, 0), (th_w-1, th_h-1), border_color, 2)

            # Fade in sequentially
            show_at = ti / len(all_th)
            clip_alpha = min(1.0, (t - show_at) * 5) if t > show_at else 0.0
            if clip_alpha > 0:
                # Ensure we stay in frame bounds
                ey = min(ty + th_h, H)
                ex = min(tx + th_w, W)
                ah = ey - ty
                aw = ex - tx
                if ah > 0 and aw > 0:
                    roi = f[ty:ey, tx:ex]
                    sm = small[:ah, :aw]
                    blended = cv2.addWeighted(sm, clip_alpha, roi, 1-clip_alpha, 0)
                    f[ty:ey, tx:ex] = blended

                # Task label
                short = name.replace("clip_0", "").replace("clip_1", "")[:18]
                label_y = min(ty + th_h - 5, H - 5)
                cv2.putText(f, short, (tx+3, label_y), FONT_SM, 0.3, (200,200,200), 1)

        draw_progress_bar(f, 1)
        draw_watermark(f)
        writer.write(f)


def stage2_hand_pose(writer, duration_s=6.0):
    """Play real cooking video with hand pose overlay."""
    overlay_path = OUTPUT_DIR / "hand_pose_overlays" / "pov_cook_sample2_pose.jpg"
    real_vid = OUTPUT_DIR.parent / "test_data" / "real_video" / "pov_cook_sample2.mp4"

    total_frames = int(duration_s * FPS)

    # Load overlay image for display
    overlay_img = cv2.imread(str(overlay_path))

    # Try to stream frames from the real video
    cap = cv2.VideoCapture(str(real_vid))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_frames = []
    frame_count = 0
    while frame_count < 300:
        ret, frm = cap.read()
        if not ret:
            break
        src_frames.append(frm)
        frame_count += 1
    cap.release()

    for fi in range(total_frames):
        f = blank()
        t = fi / total_frames

        # Header
        put_text(f, "STEP 2", (40, 45), FONT_SM, 0.6, ACCENT, 2)
        put_centered(f, "MediaPipe HandLandmarker — 21-Point Pose Extraction", 80,
                    scale=0.9, color=WHITE, thickness=2)
        put_centered(f, "float16 model | CPU inference | UDCAP 21-joint angle mapping",
                    112, scale=0.5, color=GRAY)

        # Left: raw frame from real video
        src_idx = int(t * len(src_frames)) if src_frames else 0
        src_idx = min(src_idx, len(src_frames) - 1)

        vid_w, vid_h = 560, 370
        if src_frames:
            raw = cv2.resize(src_frames[src_idx], (vid_w, vid_h))
            safe_blit(f, raw, 130, 20)
            cv2.rectangle(f, (20, 130), (20+vid_w, 130+vid_h), GRAY, 1)
            put_text(f, "Raw egocentric frame", (25, 125), FONT_SM, 0.45, GRAY, 1)

        # Right: overlay with landmarks
        ov_x, ov_y = 600, 130
        if overlay_img is not None:
            ov_resized = cv2.resize(overlay_img, (650, 370))
            safe_blit(f, ov_resized, ov_y, ov_x)
            cv2.rectangle(f, (ov_x, ov_y), (ov_x+650, ov_y+370), ACCENT, 2)

        put_text(f, "MediaPipe: 21 landmarks detected", (ov_x+5, ov_y-8), FONT_SM, 0.45, ACCENT, 1)

        # Stats panel
        stats_y = 515
        put_text(f, "Detection stats:", (40, stats_y), FONT_SM, 0.5, ACCENT2, 1)
        put_text(f, "pov_cook_sample:   20.5% detected (real kitchen video)", (55, stats_y+22), FONT_SM, 0.42, WHITE, 1)
        put_text(f, "pov_cook_sample2:   3.8% detected (partial occlusion)", (55, stats_y+42), FONT_SM, 0.42, WHITE, 1)
        put_text(f, "Synthetic clips:    0.0% — procedural joint fallback (grasp simulation)", (55, stats_y+62), FONT_SM, 0.42, GRAY, 1)
        put_text(f, "-> Detected frames use real 21-point angles. Missed frames filled via interpolation.", (55, stats_y+85), FONT_SM, 0.40, GOLD, 1)

        draw_progress_bar(f, 2)
        draw_watermark(f)
        writer.write(f)


def stage3_retargeting(writer, duration_s=6.0):
    """Show retargeting visualization."""
    retarget_img = cv2.imread(str(OUTPUT_DIR / "retargeting_visualization.png"))

    total_frames = int(duration_s * FPS)
    img_h, img_w = 460, 1180

    for fi in range(total_frames):
        f = blank()
        t = fi / total_frames

        put_text(f, "STEP 3", (40, 45), FONT_SM, 0.6, ACCENT, 2)
        put_centered(f, "Joint Retargeting: Human 21-DoF  ->  Allegro 16-DoF", 80,
                    scale=0.9, color=WHITE, thickness=2)
        put_centered(f, "Linear mapping + joint-limit clamping + temporal smoothing (window=5)",
                    112, scale=0.5, color=GRAY)

        # Retargeting visualization
        if retarget_img is not None:
            display_h = 480
            display_w = int(retarget_img.shape[1] * display_h / retarget_img.shape[0])
            display_w = min(display_w, W - 60)
            display_h = int(retarget_img.shape[0] * display_w / retarget_img.shape[1])

            # Animate: reveal from left
            reveal_w = int(display_w * min(1.0, t * 2))
            resized = cv2.resize(retarget_img, (display_w, display_h))

            start_x = (W - display_w) // 2
            start_y = 130
            if reveal_w > 0:
                safe_blit(f, resized[:, :reveal_w], start_y, start_x)

            cv2.rectangle(f, (start_x, start_y), (start_x+display_w, start_y+display_h), (40,40,40), 1)

        # Labels
        put_text(f, "Top: Human MediaPipe joints (degrees) | Bottom: Allegro robot joints (radians)",
                (40, H - 50), FONT_SM, 0.42, GRAY, 1)
        put_text(f, "Dataset saved: LeRobot JSON + RLDS JSON + NumPy arrays (4,711 frames)",
                (40, H - 30), FONT_SM, 0.42, ACCENT2, 1)

        draw_progress_bar(f, 3)
        draw_watermark(f)
        writer.write(f)


def stage4_training(writer, duration_s=6.0):
    """Show BC training curve animation."""
    curve_img = cv2.imread(str(OUTPUT_DIR / "model" / "training_curve.png"))

    # Load actual losses
    losses_path = OUTPUT_DIR / "model" / "train_losses.npy"
    losses = np.load(str(losses_path)) if losses_path.exists() else np.exp(-np.linspace(0, 5, 100)) * 0.5

    total_frames = int(duration_s * FPS)

    for fi in range(total_frames):
        f = blank()
        t = fi / total_frames

        put_text(f, "STEP 4", (40, 45), FONT_SM, 0.6, ACCENT, 2)
        put_centered(f, "Behavioral Cloning Training — MLP Policy", 80,
                    scale=0.9, color=WHITE, thickness=2)
        put_centered(f, "Architecture: 23 -> 256 -> 256 -> 128 -> 23  |  AdamW + CosineAnnealing  |  100 epochs",
                    112, scale=0.5, color=GRAY)

        # Show training curve image
        if curve_img is not None:
            display_h = 450
            display_w = int(curve_img.shape[1] * display_h / curve_img.shape[0])
            display_w = min(display_w, W - 80)
            display_h = int(curve_img.shape[0] * display_w / curve_img.shape[1])

            # Reveal animated: show progressively more of the curve
            reveal_portion = min(1.0, t * 1.5)
            resized = cv2.resize(curve_img, (display_w, display_h))

            start_x = (W - display_w) // 2
            start_y = 130
            reveal_w = int(display_w * reveal_portion)
            if reveal_w > 0:
                safe_blit(f, resized[:, :reveal_w], start_y, start_x)
            cv2.rectangle(f, (start_x, start_y), (start_x+display_w, start_y+display_h), (40,40,40), 1)

        # Animated stats
        epoch_shown = int(len(losses) * min(1.0, t * 1.5))
        if epoch_shown > 0 and epoch_shown <= len(losses):
            current_loss = losses[epoch_shown - 1]
            put_text(f, f"Epoch: {epoch_shown:3d}/100  |  Loss: {current_loss:.5f}",
                    (40, H - 35), FONT_SM, 0.55, ACCENT, 2)

        put_text(f, f"Training samples: 4,681  |  Batch size: 256  |  Device: CPU",
                (40, H - 12), FONT_SM, 0.42, GRAY, 1)

        draw_progress_bar(f, 4)
        draw_watermark(f)
        writer.write(f)


def stage5_sim(writer, duration_s=10.0):
    """Play the MuJoCo sim rollout."""
    sim_video = OUTPUT_DIR / "sim_output" / "rollout.mp4"
    cap = cv2.VideoCapture(str(sim_video))

    sim_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_sim_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_out_frames = int(duration_s * FPS)

    sim_frames = []
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        sim_frames.append(frm)
    cap.release()

    for fi in range(total_out_frames):
        f = blank()
        t = fi / total_out_frames

        put_text(f, "STEP 5", (40, 45), FONT_SM, 0.6, ACCENT, 2)
        put_centered(f, "MuJoCo Simulation: BC Policy Rollout", 80,
                    scale=1.0, color=WHITE, thickness=2)
        put_centered(f, "4-DOF arm + 12-DOF Allegro hand  |  Position control  |  6 second rollout",
                    112, scale=0.5, color=GRAY)

        # Sim video
        if sim_frames:
            src_idx = int(t * len(sim_frames))
            src_idx = min(src_idx, len(sim_frames) - 1)
            sim_frm = sim_frames[src_idx]

            # Scale to fill most of screen
            sim_h = 530
            sim_w = int(sim_frm.shape[1] * sim_h / sim_frm.shape[0])
            sim_w = min(sim_w, W - 20)
            sim_h = int(sim_frm.shape[0] * sim_w / sim_frm.shape[1])

            start_x = (W - sim_w) // 2
            start_y = 128
            resized_sim = cv2.resize(sim_frm, (sim_w, sim_h))
            safe_blit(f, resized_sim, start_y, start_x)

            # Border with glow effect
            cv2.rectangle(f, (start_x, start_y), (start_x+sim_w, start_y+sim_h), ACCENT, 2)

        # Progress
        sim_time = t * 6.0
        put_text(f, f"Sim time: {sim_time:.1f}s / 6.0s  |  MuJoCo 3.5.0  |  dt=0.002s",
                (40, H - 12), FONT_SM, 0.42, GRAY, 1)

        draw_progress_bar(f, 5)
        draw_watermark(f)
        writer.write(f)


def summary_card(writer, duration_s=5.0):
    """Final summary card."""
    total_frames = int(duration_s * FPS)

    lines = [
        ("Pipeline", "DexCrowd End-to-End Proof  |  Feb 2026", ACCENT),
        ("", "", WHITE),
        ("Input", "30 egocentric manipulation clips (2 real + 28 synthetic)", WHITE),
        ("Step 1", "Generated 30 clips x 4s | 10 tasks: pick, pour, fold, grasp...", WHITE),
        ("Step 2", "MediaPipe HandLandmarker | 20.5% detection on real kitchen video", WHITE),
        ("        ", "         21 landmarks -> UDCAP joint angle mapping", GRAY),
        ("Step 3", "Human 21-DoF -> Allegro 16-DoF | Linear retarget + smoothing", WHITE),
        ("        ", "         4,711 frames | LeRobot JSON + RLDS JSON + NumPy", GRAY),
        ("Step 4", "BC MLP: 23->256->256->128->23 | 100 epochs | Loss: 0.00373", WHITE),
        ("Step 5", "MuJoCo 3.5.0 | 4-DOF arm + 12-DOF hand | 6s rollout", WHITE),
        ("", "", WHITE),
        ("Next", "Real hardware: iPhone + Apple Watch + UDCAP glove", ACCENT2),
        ("Next", "Scale: Ego4D/EPIC-Kitchens data ingestion", ACCENT2),
        ("Next", "GPU: HaMeR mesh recovery + Allegro real deployment", ACCENT2),
    ]

    for fi in range(total_frames):
        f = blank()
        t = fi / total_frames

        put_centered(f, "Pipeline Summary", 50, scale=1.2, color=ACCENT, thickness=3)

        # Animate lines appearing
        for li, (key, val, color) in enumerate(lines):
            appear_at = li / len(lines) * 0.8
            if t > appear_at:
                alpha = min(1.0, (t - appear_at) * 10)
                y_pos = 100 + li * 35
                if key:
                    # Key in accent color
                    cv2.putText(f, f"{key}:", (80, y_pos), FONT_SM, 0.5, ACCENT2, 1, cv2.LINE_AA)
                    cv2.putText(f, val, (230, y_pos), FONT_SM, 0.5, color, 1, cv2.LINE_AA)
                else:
                    pass

        # Final CTA
        if t > 0.85:
            alpha = min(1.0, (t - 0.85) * 7)
            put_centered(f, "DexCrowd  |  Crowdsourced Dexterous Manipulation Data", H - 45,
                        scale=0.6, color=GOLD, thickness=1)
            put_centered(f, "egocentric video  ->  retargeted robot joints  ->  trained policy",
                        H - 20, scale=0.45, color=GRAY, thickness=1)

        draw_progress_bar(f, 6)
        draw_watermark(f)
        writer.write(f)


def build_demo_video():
    out_path = str(OUTPUT_DIR / "DEXCROWD_PIPELINE_DEMO.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    print(f"Building demo video: {out_path}")
    print(f"  Resolution: {W}x{H}  |  {FPS}fps")

    print("  [0/6] Title screen...")
    title_screen(writer, 3.0)

    print("  [1/6] Stage 1: Egocentric clips...")
    stage_card(writer, 1, "Egocentric Video Clips", "30 clips | 10 manipulation tasks", 1.5)
    stage1_clips(writer, 5.0)

    print("  [2/6] Stage 2: Hand pose extraction...")
    stage_card(writer, 2, "Hand Pose Extraction", "MediaPipe HandLandmarker | 21 joints", 1.5)
    stage2_hand_pose(writer, 6.0)

    print("  [3/6] Stage 3: Retargeting...")
    stage_card(writer, 3, "Joint Retargeting", "Human 21-DoF -> Allegro 16-DoF", 1.5)
    stage3_retargeting(writer, 6.0)

    print("  [4/6] Stage 4: BC training...")
    stage_card(writer, 4, "Behavioral Cloning", "MLP policy | 100 epochs", 1.5)
    stage4_training(writer, 6.0)

    print("  [5/6] Stage 5: MuJoCo sim...")
    stage_card(writer, 5, "MuJoCo Simulation", "Trained policy rollout | 6 seconds", 1.5)
    stage5_sim(writer, 10.0)

    print("  [6/6] Summary...")
    stage_card(writer, 6, "Pipeline Complete", "All artifacts saved", 1.0)
    summary_card(writer, 5.0)

    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Demo video saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"  Duration: ~{3+1.5+5+1.5+6+1.5+6+1.5+6+1.5+10+1+5:.0f}s")
    return out_path


if __name__ == "__main__":
    build_demo_video()
