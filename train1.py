from ultralytics import YOLO

model = YOLO("yolov8s.pt")


dataset_path = "/home/iclab/yolo_train/src/HSU1/dataset/dataset1/data.yaml"

model.train(
    data=dataset_path,      # 数据集配置文件路径
    epochs=100,             # 总训练轮数
    imgsz=640,              # 输入图像尺寸
    batch=8,                # batch size（依 GPU 内存大小调整）
    optimizer="AdamW",      # 使用 AdamW 优化器
    lr0=0.0005,             # 初始学习率
    lrf=0.01,               # 最终学习率比例
    weight_decay=0.0005,    # 权重衰减系数
    momentum=0.937,         # 动量参数
    device="cuda",          # 使用 GPU 训练p
    workers=8,              # 数据加载进程数
    amp=True,               # 启用自动混合精度训练
    patience=50,            # 50 轮无提升提前停止
    val=True,               # 进行验证
    save=True,              # 保存模型
    save_period=10,         # 每 10 轮保存一次
    cache="ram",            # 仅缓存训练数据，节省内存
    dropout=0.2,            # Dropout 降低至 0.2，减少信息丢失
    mosaic=0.9,             # Mosaic 数据增强
    mixup=0.2,              # 降低 mixup，减少噪声干扰
    copy_paste=0.3,         # 保持 copy-paste 数据增强
    overlap_mask=True,      # 启用重叠 mask 计算
    mask_ratio=4,           # mask loss 计算比例参数
    hsv_h=0.02, hsv_s=0.8, hsv_v=0.6,  # 提高 HSV 颜色增强范围
    iou=0.35,               # 降低 IoU 阈值减少漏检
    project="runs/segment", # 结果输出目录
    name="train_m_optimized",  # 训练实验名称
    fliplr=0,
    flipud=0,
    degrees=0
)
