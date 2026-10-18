# 进食区间检测：ActionFormer 基线

实现依据 `00 TaskRequirments/#U63a5#U53e3#U8bf4#U660e.docx`（接口说明）、`#U6d4b#U8bd5#U8bf4#U660e.docx`（测试说明）及随附的 input.py/output.py。官方两个接口文件保持原样。

## 当前状态

已经在真实数据的 6 文件调试子集上完成特征准备、5 epoch 训练、验证、独立测试、保存/加载模型和比赛格式推理。它只用于确认流程：验证/测试事件 F1=0，默认阈值未检出进食，不能作为有效提交模型。尚未完成全量训练。没有在 Windows 机器上构建或验证 EXE。

本机可用 Python：

```bash
/seu_share2/home/213222097/.conda/envs/meb_env/bin/python
```

该环境为 Python 3.10.6、PyTorch 2.1.2，无可用 CUDA。系统默认 python 是 3.6，不能用来运行本流程。其他机器请使用 Python >=3.9，安装 `requirements-intake.txt`；GPU 机器按其 CUDA 版本安装匹配的 torch。

## 数据流与标签

```text
Data/IMU+PPG.csv → 原始 TXT → 各模态按绝对时间独立汇总 → 带时间坐标的特征序列
                                      ↓
Data/MealInfo.csv → 同 healthid 且同 externalid 的用餐区间 → ActionFormer 分类+边界回归
                                                                ↓
                                             开始毫秒、结束毫秒、置信度
```

- 只按明确作废 externalid 排除 HNU21007、HNU21026j、HNU21030，不扩展到同 healthid 的其他编号；再次读取 Data/excluded_subjects.json 防止错误回流。
- 训练只接受 HNU 数字（可带一个字母后缀）的编号。test、空编号、明显残缺编号记录进跳过清单，不自动猜测身份。
- 用餐标签使用 `timeFrame.startTime/endTime`，不使用上传时间。现有 278 条记录中，有 22 条与 beforeTime/afterTime 不同；后者还存在起止相等或倒序的情况，因此基线明确采用 timeFrame，并把差异保存为 label_field_disagreements.json。输出列名叫 beforeTime/afterTime，不代表训练表同名字段一定是主办方最终金标准，这一点仍需核实。标签关联为 `healthid AND 标准化后的 externalid`，避免相同账号不同编号被擅自合并。
- 已标注账号的采集区间外暂作背景，依赖“用餐记录完整”的假设；漏标会污染背景。训练清单中明确记录此假设。
- 划分训练/验证/测试时，将共享 healthid 或 externalid 的记录组成连通分组，保证两种标识都不跨集合；这只是防止泄漏的保守分组规则，不用来扩展作废范围或标签。
- 归一化参数只在训练集拟合，并随检查点保存。验证集选 best.pt，测试集不参与模型选择。

## 输入特征及变长记录

默认每 **5 秒**汇总一次，形成 **67 × T** 特征：

- ACC 三轴和 GYRO 三轴：带符号 log1p 压缩后，各计算 mean/std/min/max，共 24 维。
- PPG1～PPG20：相同变换后，各计算 mean/std，共 40 维。通道含义未知，当前只是基线选择，可通过配置修改数量。
- ACC/GYRO/PPG 三个存在性指示，共 3 维。全零模态行和无效时间戳不进入统计；缺失模态的统计值填零，存在性为 0。PPG 并未被转换成心率。

重复的批量时间戳不会被去重删样本，也没有假定原始采样频率。各模态使用自己的 13 位时间戳分箱，再按相同时间格对齐。真实采样间隔、原始物理单位和批量时间戳规则仍需设备文档核实。

保留完整的有效连续记录，不人为裁出“吃饭片段”训练，也不把整个 TXT 标成一个类别。ACC 覆盖的时间格之间超过 30 秒时分成不同有效运行段，避免模型跨长时间缺口定位。较短缺失格保留为零值和缺失指示；没有跨缺口插值。实际可输出边界裁剪到 ACC 观测范围。

ActionFormer 使用原始 `Actionformer/libs/modeling/meta_archs.py` 中注册的 `LocPointTransformer`，保留局部 Transformer、多尺度特征金字塔、focal 分类损失与 DIoU 边界回归。只调整传感器输入维数、单一动作类别和模型宽度。批量训练用原网络的 padding/mask。

特征中心位于 `grid_start_ms + (index+0.5)*bin_ms`。训练区间坐标为 `(timestamp_ms-grid_start_ms)/bin_ms-0.5`；预测使用原模型的中心偏移转换后再加回绝对起点。不会把相对秒数或特征序号直接写进结果。

默认训练长度上限 16384 个时间格（约 22.8 小时）。准备程序会把超限有效段列入 manifest 的 rejected_runs，绝不截断。全量训练前必须检查这张清单；若有超限段，应调大 max_tokens 并核查数据/显存。调试子集没有超限段。推理会扩展非持久化点坐标缓存以处理更长序列，但仍受机器内存限制。

## 命令

在项目根目录执行；下面的 `python` 指 Python >=3.9 的目标环境。

```bash
# 全量准备：读取全部符合条件的采集文件，保留原始文件
python -m intake.prepare --output artifacts/intake

# 全量训练：GPU 可改为 --device cuda:0
python -m intake.train --cache artifacts/intake --output runs/actionformer --config experiment261016/intake_actionformer.yaml --device cpu

# 独立测试集评估
python -m intake.evaluate --cache artifacts/intake --checkpoint runs/actionformer/best.pt --split test

# 单元和接口集成测试
python -m unittest discover -s tests -v
```

准备阶段可能需要读取近 100 GB 原始 TXT，耗时明显长于调试。输出目录不能包含旧 manifest，避免缓存混用。配置在 `experiment261016/intake_actionformer.yaml`（兼容旧 JSON），包含时间格、PPG 数量、训练超参数和推理阈值。

复现最小流程时使用全新的目录：

```bash
python -m intake.prepare --output artifacts/my_smoke --limit-files 6
python -m intake.train --cache artifacts/my_smoke --output runs/my_smoke --epochs 5 --device cpu
python -m intake.evaluate --cache artifacts/my_smoke --checkpoint runs/my_smoke/best.pt --output runs/my_smoke/test_metrics.json
python -m intake.make_smoke_fixture --cache artifacts/my_smoke --output artifacts/my_fixture/ACC_test_data
python submission.py --data-root artifacts/my_fixture/ACC_test_data --checkpoint runs/my_smoke/last.pt --output artifacts/my_fixture/results.csv
```

调试子集有意优先选择不同账号中较短且索引时间与用餐重叠的文件，以覆盖边界回归分支。它不是有代表性的验证/测试集；manifest 标记 debug_subset=true。实际信号时间与索引可能不一致，因此准备后仍可能出现无正例文件。

根据用户提供的评测规则，主指标为事件 F1，严格以 **IoU > 0.25** 判定匹配成功（等于 0.25 不算）。同时报告 sensitivity=TP/(TP+FN)（灵敏度）、positive_predictive_value=TP/(TP+FP)（阳性预测率）、f1=2TP/(2TP+FP+FN)。先在每位参与者内按预测置信度从高到低，与尚未匹配的最大 IoU 真实事件一对一匹配，再汇总所有参与者的 TP/FP/FN 计算指标；同一真实事件不能被重复计为 TP。

次指标仅在正确匹配事件上计算：start_mae_seconds 为开始时间 MAE，end_mae_seconds 为结束时间 MAE；boundary_mae_seconds 为二者平均。所有 MAE 单位为秒，无 TP 时记为 null，而不是 0。无预测或无真值时，未定义的灵敏度/阳性预测率及 F1 采用 0。训练每轮报告全部指标，以验证集 F1 最大保存 best.pt；F1 相同时用更低的 boundary_mae_seconds 决胜。测试集不参与模型选择。YAML 的 evaluation.iou_threshold 控制评估阈值，与 inference.nms_iou（候选去重阈值）含义不同。

本地评估仍限于实际保留信号范围内的标签，不包含无法观测的用餐区间。用户给出的公式没有规定重复候选的匹配细节，上述置信度优先一对一匹配是当前明确采用的实现约定。此前调试训练日志使用 IoU>=0.5，保留为历史记录，不能与新指标直接比较。

## 比赛推理接口

```text
运行目录/
  IntakeDetection.exe 或 submission.py
  model.pt
  ACC_test_data/
    userinfobean.txt
    HNU00001/
      collect_data0_....txt
      collect_data1_....txt
    HNU00002/
      ...
  results.csv  ← 本轮生成
```

```bash
python submission.py --data-root /path/to/ACC_test_data --checkpoint /path/to/model.pt --output /path/to/results.csv
```

- 用官方 `InfoReader(...).read_info(编号)` 读取每人的资料；当前基线暂不使用人口学字段作为模型特征。
- 用官方 `DataReader(个人目录).iter_data()` 遍历个人目录直接包含的全部 TXT，每次处理一个完整文件。不依赖训练索引，不读取测试金标准。
- 只将预测进食区间传给官方 `save_result(..., 1)`。同一人的重叠文件候选按绝对时间执行 NMS；不会按文件名顺序推断时间。
- 每次运行从空的临时结果开始，完成后原子替换 results.csv。不会追加旧结果。无预测也有表头 `externalid,beforeTime,afterTime`，无额外列。
- start/end 为 13 位毫秒整数且 start<end。写完调用 `validate_result` 校验格式。
- 空 TXT 或没有有效 ACC 的文件不产生区间。缺少 ACC 必要字段、非法文本格式或其他处理失败会明确报错，不伪造成功结果。
- 默认离线运行，需要完整文件后预测；不属于实时/因果检测。

## Windows 打包

必须在 **Windows** 上用匹配环境构建 Windows EXE；Linux 上跑通 Python 不代表已完成 Windows 验收。

```powershell
python -m pip install -r requirements-intake.txt
python -m pip install pyinstaller
pyinstaller intake_submission.spec
Copy-Item runs/actionformer/best.pt dist/IntakeDetection/model.pt
```

将 `dist/IntakeDetection/` 整个目录（EXE、model.pt 和 `_internal` 依赖目录）打包，不能只拷贝 EXE。EXE 默认以自身所在目录定位 ACC_test_data、model.pt 和 results.csv，支持从其他工作目录启动。官方 input.py/output.py 作为资源包含。

为减少 Windows 编译依赖，对 ActionFormer 的可选 C++ NMS 增加了纯 PyTorch CPU 回退（nms_fallback.py）；安装了扩展时仍优先使用扩展。当前基线使用 hard NMS。

提交前需要在另一台 Windows 电脑验证：多人多文件、空 TXT、全无进食、重复启动覆盖旧结果、毫秒时间戳、CPU 环境加载模型。邮件提交属于最终发布步骤，本实现没有发送邮件。

## 已生成的调试产物

- `artifacts/smoke_v2/manifest.json`：真实 6 文件特征、账号划分、跳过清单及归一化参数。
- `runs/actionformer_smoke/{best.pt,last.pt,history.json,test_metrics.json}`：5 epoch 调试训练产物。
- `artifacts/competition_smoke/results.csv`：官方目录推理生成的仅含表头结果。
- `reports/actionformer_smoke_verification.json`：实际检查与运行环境记录。

数据准备发现现有 1150 个 TXT 中 1134 个以 ACC_TIME 表头开头，另 16 个是非表格二进制内容；它们未删除，并列入跳过清单。在明确原格式或修复之前，不将其当成正常文本信号。旧 reports 中的原始规模统计不代表本流程最终训练规模。

## Slurm 提交作业

YAML 配置：`experiment261016/intake_actionformer.yaml`；提交脚本：`submit_actionformer.sh`。

```bash
# 项目根目录执行；只在此命令执行后才真正入队
sbatch submit_actionformer.sh

# 使用其他分区或配置文件
sbatch -p <分区名> submit_actionformer.sh /绝对路径/实验配置.yaml
```

脚本默认申请 luc 分区、1 张 GPU、4 个 CPU、24 小时；内存由集群按 CPU 核数自动匹配，不能通过 --mem 显式申请。luc 取自已有作业脚本，当前未向调度器查询可用性，可用 sbatch 参数覆盖。Python 指向已验证的 meb_env 环境，也可通过 INTAKE_PYTHON 指定其他环境。

作业调用 `intake.run_job`，依次准备/检查缓存、按 YAML 训练、用 best.pt 在测试集上评估。训练/测试结果保存在 `runs/actionformer/job_<作业编号>/`；日志在项目根目录 `slurm-intake-actionformer-<作业编号>.out/.err`。这里的 output_dir 基路径、数据路径、特征缓存路径和设备均可在 YAML 的 run 段修改。

已有缓存只有预处理参数、划分种子、长度上限、调试/全量模式及调试文件数匹配时才复用，否则报错并要求换 cache_dir。发现超限运行段时作业也会停止，避免直接训练不完整的集合。缓存不自动检测原 CSV 后续变更；更改原数据或排除规则后，应使用新 cache_dir。多个作业共享缓存时使用文件锁防止并行准备冲突。重复运行不覆盖已有 last.pt。

调试：把 YAML 中 run.limit_files 改成 6、train.epochs 改成 5，并换一个 cache_dir。恢复全量时设置 limit_files=0 并换回全量缓存。GPU 不可用时明确报错，不会自动切换 CPU；手动 CPU 验证可以修改 run.device。

## TriDet 实验

TriDet 的配置放在 `experiment261016/intake_tridet.yaml`，提交脚本为 `submit_tridet.sh`：

```bash
sbatch submit_tridet.sh experiment261016/intake_tridet.yaml
```

不传参数时也默认读取该 YAML。同一 SH 可以提交其他兼容配置；网络由 YAML 的 model.name 选择，而不是由 SH 名称决定。旧配置没有 model.name 时默认 ActionFormer。

TriDet 实际使用 `TriDet-master/libs/modeling/meta_archs.py` 的原版 TriDet，包含 SGPBackbone 和 Trident 边界分布检测头；两个仓库的同名 libs 包在独立命名空间加载，避免误调用 ActionFormer。模型宽度需要能被 16 整除。SGP 参数 sgp_window、sgp_mlp_dim、num_bins、k、iou_weight_power、input_noise 在 YAML 中设置。

默认申请 luc 分区一张 GPU，训练 30 epoch。结果写入 `runs/tridet/job_<作业编号>/`；日志为 `slurm-intake-tridet-<作业编号>.out/.err`。使用相同的 meb_env、67 维传感器特征、账号划分、作废排除规则、严格 IoU>0.25 的事件 F1 和匹配事件 MAE。best.pt 同样按验证集 F1 选择，F1 相同时比较平均边界 MAE。默认可复用 ActionFormer 的 `artifacts/intake` 全量缓存，前提是预处理、划分种子、长度上限及数据内容相同。

单独训练命令（需要显式传 TriDet 配置，不能仅依赖共享缓存中的模型配置）：

```bash
python -m intake.train --cache artifacts/intake --config experiment261016/intake_tridet.yaml --output runs/tridet --device cuda:0
```

已在 CPU 上完成真实 6 文件调试缓存的一轮训练，并验证正负样本反向传播、模型保存/加载和推理。调试产物在 `runs/tridet_smoke/`，不是全量训练结果或有效比赛模型。GPU 训练尚未验证。TriDet 的 NMS 也提供与 ActionFormer 相同的纯 PyTorch CPU 回退实现，无需先编译 C++ 扩展。
