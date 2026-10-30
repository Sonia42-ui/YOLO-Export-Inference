# -*- coding: utf-8 -*-
"""
export.py —— 把 PyTorch 权重 models/best.pt 导出成 ONNX 模型 models/model.onnx
================================================================================
用法：
    python export.py

导出配置（和训练时保持一致）：
    * 输入尺寸 640x640
    * ONNX opset 12（兼容性最好：onnxruntime / OpenCV DNN / TensorRT 都支持）
    * 静态输入形状（dynamic=False）
    * 用 onnxslim 精简计算图（去掉可以合并的冗余节点）
================================================================================
"""

from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------- 路径与参数
PROJECT_DIR = Path(__file__).resolve().parent
WEIGHT = PROJECT_DIR / "models" / "best.pt"        # 原始 PyTorch 权重
OUTPUT = PROJECT_DIR / "models" / "model.onnx"     # 导出的 ONNX 模型

IMGSZ = 640        # 输入尺寸，必须和训练时一致
OPSET = 12         # ONNX 算子集版本
SIMPLIFY = True    # 用 onnxslim 精简计算图
DYNAMIC = False    # 固定输入形状，部署最稳


def main() -> None:
    # 1. 检查权重是否存在
    if not WEIGHT.exists():
        raise FileNotFoundError(f"找不到权重文件：{WEIGHT}")

    print(f"[1/3] 加载权重：{WEIGHT}")
    model = YOLO(str(WEIGHT))
    # 打印模型基本信息：任务、类别、关键点形状
    print(f"      任务={model.task}  类别={model.names}")
    print(f"      关键点={model.model.kpt_shape}  参数量={sum(p.numel() for p in model.model.parameters()):,}")

    # 2. 切到推理模式（关闭 Dropout / BatchNorm 的训练行为），导出必须做
    model.model.eval()

    # 3. 导出 ONNX
    #    ultralytics 内部：造一个假的 640x640 输入跑一遍前向，
    #    用 torch.onnx.export 把每个算子"录制"成 ONNX 节点，再合并/精简，
    #    最后在 best.pt 旁边生成 best.onnx
    print(f"[2/3] 开始导出 ONNX（imgsz={IMGSZ}, opset={OPSET}, simplify={SIMPLIFY}）...")
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        simplify=SIMPLIFY,
        dynamic=DYNAMIC,
        device="cpu",      # 导出只是"翻译"，用 CPU 更稳定
    )

    # 4. 把 ultralytics 生成的 best.onnx 改名成 model.onnx
    tmp = Path(exported)
    if tmp.resolve() != OUTPUT.resolve():
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(OUTPUT)          # 覆盖旧的 model.onnx

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"[3/3] 导出完成：{OUTPUT}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
