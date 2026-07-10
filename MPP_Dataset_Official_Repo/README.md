# MPP Dataset Official Repo

本目录用于归档论文发布版本对应的正式代码。仓库仅保留代码、依赖和说明；数据划分 CSV、实验输出、混淆矩阵、模型权重、审计表格和 metadata 快照不作为代码仓库内容保留。

## 论文对应关系

- `preprocessing/`
  - `data_to_png_pipeline.py`：论文数据预处理主流水线。对应 DICOM 元数据解析、Rescale Slope/Intercept HU 还原、B-Spline 1.0 mm 等向重采样、HU `[-1000, -400]` 肺部轴向范围识别、肺窗 WL=-600/WW=1500 映射和 8-bit PNG 导出。
  - `examples/standardize_dataset.py`：厚层/薄层病例标准化示例脚本，用于演示重采样后的空间标准化效果。

- `data_integrity/`
  - `verify_patientid_dirs.py`：DICOM 完整性、PatientID 一致性、非 DICOM 文件和损坏文件核查。
  - `filter_slice_thickness.py`：论文 “严格仅保留原始重建层厚 <= 1.5 mm” 的代码依据。读取 mapping/metadata 表中的 `SliceThickness` 字段，默认 dry-run 标记 `>1.5 mm` 序列；仅在指定 `--execute --archive-root` 时隔离数据。

- `qa_pipeline/`
  - `lung_area_audit.py`：论文 Technical Validation 中解剖感知切片筛查的主要代码依据之一。实现暗区面积比例 8% 与主连通域面积 1% 的非肺部切片审计；当前代码使用固定暗区阈值 `pixel < 40`，不是完整 Otsu 实现。
  - `lightweight_filter.py`：保守型轻量后处理脚本，用于头尾边缘切片和极低肺区面积切片的补充清洗；不作为论文 Otsu 表述的直接实现。
  - `motion_artifact_filter.py`：基于 Laplacian variance 的运动伪影、重影和模糊切片审计；默认 dry-run。
  - `png_quality_check.py`：只读 PNG 质量检查脚本，用于标记读取失败、极黑、低纹理等疑似无效切片。

- `classification/`
  - `data_split.py`：按患者级别生成训练、验证、测试划分，支持论文中的 7:1:2 patient-level split。
  - `models.py`：定义 ViT-B/16、ResNet-50、ConvNeXt-Tiny、Swin-Tiny 等基线模型。
  - `train.py`：训练入口。对应预训练权重初始化、AdamW、50 epochs 和 WeightedRandomSampler。
  - `evaluate.py`：评估入口。生成 accuracy、macro precision/recall/F1 和归一化混淆矩阵。
  - `dataset.py`：读取 PNG 路径和 split CSV，构建训练/验证/测试数据集。

- `audit_and_reporting/`
  - `final_dataset_deep_audit.py`：统计患者数、序列数、切片数、像素均值/标准差、空目录和损坏文件。
  - `generate_markdown_report.py`：根据审计 JSON 生成 Markdown 报告。

- `figures/`
  - `plot_paper_figs.py`：代表性切片、连续切片和论文图形生成工具。

## 推荐复现流程

1. 核查原始 DICOM 完整性和 PatientID 一致性：
   ```bash
   python data_integrity/verify_patientid_dirs.py
   ```

2. 根据 metadata/mapping 表审计并排除层厚 `>1.5 mm` 的序列：
   ```bash
   python data_integrity/filter_slice_thickness.py \
     --mapping-csv /path/to/mapping.csv \
     --out-csv reports/removed_over_1.5mm.csv
   ```

3. 将 DICOM 序列转换为标准化肺窗 PNG：
   ```bash
   python preprocessing/data_to_png_pipeline.py \
     --source-config configs/source_dirs.example.json \
     --output-root /path/to/png_dataset
   ```

4. 执行 PNG 质量、肺区连通域和运动伪影审计：
   ```bash
   python qa_pipeline/png_quality_check.py --data-root /path/to/png_dataset --out-csv reports/png_quality_audit.csv
   python qa_pipeline/lung_area_audit.py --data-root /path/to/png_dataset --out-csv reports/lung_area_audit.csv
   python qa_pipeline/motion_artifact_filter.py --data-root /path/to/png_dataset --out-csv reports/motion_artifact_audit.csv
   ```

5. 生成患者级训练/验证/测试划分：
   ```bash
   python classification/data_split.py \
     --data_root /path/to/png_dataset \
     --out_dir work/splits
   ```

6. 训练和评估基线模型：
   ```bash
   python classification/train.py \
     --data_root /path/to/png_dataset \
     --split_csv work/splits/slices_split.csv \
     --task A \
     --model swin_tiny \
     --epochs 50 \
     --out_dir work/outputs

   python classification/evaluate.py \
     --data_root /path/to/png_dataset \
     --split_csv work/splits/slices_split.csv \
     --task C \
     --out_dir work/outputs
   ```

7. 生成数据集审计报告：
   ```bash
   python audit_and_reporting/final_dataset_deep_audit.py \
     --data-root /path/to/png_dataset \
     --out-json reports/dataset_global_stats.json

   python audit_and_reporting/generate_markdown_report.py \
     --stats-json reports/dataset_global_stats.json \
     --out-md reports/dataset_audit_report.md
   ```

## 安全默认值

- 过程审计脚本默认不移动、不删除数据。
- 需要实际隔离文件时，必须显式指定 `--execute --archive-root`。
- 本仓库不内置本机绝对路径；数据根目录、输出目录和 split CSV 均由命令行参数提供。
