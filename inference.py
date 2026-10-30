# -*- coding: utf-8 -*-
"""
inference.py —— 加载模型（.pt 或 .onnx）对视频逐帧推理，输出带标注的结果视频
================================================================================
用法：
    # 用导出的 ONNX 模型推理（默认）
    python inference.py

    # 用原始 PyTorch 模型推理
    python inference.py --model models/best.pt --output results/pt_result.mp4

    # 指定视频、阈值、只跑前 100 帧
    python inference.py --source videos/experiment_1.mp4 --conf 0.25 --iou 0.7 --max-frames 100

两种模型的预处理、后处理、NMS、画图都交给 ultralytics 同一套代码完成，
因此 .pt 与 .onnx 的推理结果可以直接对比。
================================================================================
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
# 默认用 CPU 推理，避免 ultralytics 在有显卡的机器上去自动安装 onnxruntime-gpu；
# 需要显卡时删掉这一行，并给 --device 传 0。

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------- 默认配置
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "model.onnx"      # 默认用导出的 ONNX 模型
DEFAULT_SOURCE = PROJECT_DIR / "videos" / "experiment_1.mp4"
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "onnx_result.mp4"

CONF = 0.25     # 置信度阈值
IOU = 0.7       # NMS 的 IoU 阈值
IMGSZ = 640     # 推理输入尺寸（与导出时一致）


def to_h264(tmp_mp4: Path, src_video: Path, out_mp4: Path) -> None:
    """
    用 ffmpeg 把 OpenCV 写出的 mp4v 临时视频转成 H.264（并保留原视频音轨）。
    理由：mp4v（MPEG-4 Part 2）部分播放器不认；H.264 + yuv420p 兼容性最好。
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        tmp_mp4.replace(out_mp4)          # 没装 ffmpeg 就直接用 mp4v 结果
        return

    cmd = [
        ffmpeg, "-y",
        "-i", str(tmp_mp4),               # 输入1：画好框的视频
        "-i", str(src_video),             # 输入2：原视频（取音轨）
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_mp4.unlink(missing_ok=True)


def infer_video(model_path, source, output, conf=CONF, iou=IOU, imgsz=IMGSZ,
                device="cpu", max_frames=0, preview_frame=0, preview_path=None) -> dict:
    """
    对视频逐帧推理并保存结果视频，返回统计信息。

    返回 dict：frames / det_total / frames_with_det / class_counts /
               elapsed / fps / model_size_mb / output
    """
    model = YOLO(str(model_path))                  # 自动识别 .pt 或 .onnx
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = w - w % 2, h - h % 2                  # H.264 要求偶数尺寸

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name("_tmp_" + output.name)

    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w2, h2))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{tmp}")

    names = model.names
    det_total = frames = frames_with_det = 0
    class_counts = {}
    frame_times = []                               # 每帧耗时，用于算中位数
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_frame = time.perf_counter()
        frames += 1
        if (frame.shape[1], frame.shape[0]) != (w2, h2):
            frame = cv2.resize(frame, (w2, h2))

        result = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz,
                               device=device, verbose=False)[0]

        n = 0 if result.boxes is None else len(result.boxes)
        if n:
            frames_with_det += 1
            det_total += n
            for c in result.boxes.cls.cpu().numpy().astype(int):
                name = names[int(c)] if isinstance(names, dict) else str(c)
                class_counts[name] = class_counts.get(name, 0) + 1

        writer.write(result.plot())                 # 画框/类别/置信度/关键点

        if preview_path and frames - 1 == preview_frame:
            cv2.imwrite(str(preview_path), result.plot())

        frame_times.append((time.perf_counter() - t_frame) * 1000)
        if max_frames and frames >= max_frames:
            break

    elapsed = time.perf_counter() - t0
    cap.release()
    writer.release()

    to_h264(tmp, Path(source), output)             # 转 H.264 + 音轨

    return {
        "frames": frames,
        "det_total": det_total,
        "frames_with_det": frames_with_det,
        "class_counts": class_counts,
        "elapsed": elapsed,
        "fps": frames / max(elapsed, 1e-6),
        "ms_median": float(np.median(frame_times)) if frame_times else 0.0,
        "model_size_mb": Path(model_path).stat().st_size / 1024 / 1024,
        "output": output,
        "video_fps": fps,
        "video_size": (w, h),
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 .pt 或 .onnx 模型，对视频做推理并保存结果视频")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="模型路径（.pt 或 .onnx）")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="输入视频路径")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出视频路径")
    parser.add_argument("--conf", type=float, default=CONF, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=IOU, help="NMS 的 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="推理输入尺寸")
    parser.add_argument("--device", default="cpu", help="cpu / 0 / 1 / auto")
    parser.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"模型：{args.model}")
    print(f"视频：{args.source}")
    stats = infer_video(args.model, args.source, args.output,
                        conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                        device=args.device, max_frames=args.max_frames)

    print(f"帧数    ：{stats['frames']}（视频共 {stats['total']} 帧，{stats['video_fps']:.2f} FPS）")
    print(f"有目标帧：{stats['frames_with_det']}    检出目标：{stats['det_total']}")
    if stats["class_counts"]:
        print("类别统计：" + "，".join(f"{k}={v}" for k, v in sorted(stats["class_counts"].items())))
    print(f"耗时    ：{stats['elapsed']:.2f}s（{stats['fps']:.1f} 帧/秒）")
    print(f"结果视频：{stats['output']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
