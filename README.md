# KataGo-PyTorch

用 **Python + PyTorch** 复现 Wu (2020) *Accelerating Self-Play Learning in Go* 中的 KataGo 训练与推理系统。
纯自我对弈、从随机权重开始、不使用任何人类棋谱，包含论文里的全部关键技巧，并配套一个可以离线运行的
NumPy 参考后端用于验证。

> 论文：David J. Wu, *Accelerating Self-Play Learning in Go*, arXiv:1902.10565 (2020)
> 前作：Silver et al., *Mastering the game of Go without human knowledge*, Nature 550 (2017)（即 AlphaGo Zero）

---

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 项目结构](#2-项目结构)
- [3. 系统架构](#3-系统架构)
- [4. 与论文的逐项对应](#4-与论文的逐项对应)
- [5. 关键实现细节](#5-关键实现细节)
- [6. 与论文的偏差（必读）](#6-与论文的偏差必读)
- [7. 测试与验证](#7-测试与验证)
- [8. 演示结果](#8-演示结果)
- [9. 性能](#9-性能)

---

## 1. 快速开始

### 环境

```bash
python >= 3.10
numpy >= 1.24
torch >= 2.3        # 可选；装了就走 torch 后端（NumPy 2.x 需要 torch 2.3+，2.0/2.1 会报 "Numpy is not available"）
```

```bash
pip install -r requirements.txt      # 或者只装 numpy，用 --backend numpy
```

项目**不需要安装**即可运行（`conftest.py` 和脚本自己把根目录加进 `sys.path`）。

### 训练

```bash
# 快速 CPU 演示（约 20 分钟，9x9 棋盘，NumPy 后端）
# §8 的结果就是这组配置跑出来的；那份记录是规则引擎优化之前测的，当时 32.5 分钟
python scripts/train.py --backend numpy --size 9 --blocks 2 --channels 16 \
    --iterations 30 --games 4 --steps 50 --batch-size 32 --lr 6e-3 \
    --lr-warmup-samples 500 --full-cap 48 --fast-cap 12 --full-fraction 0.25 \
    --min-pass-move 40 --replay-window 100000 --no-gating \
    --eval-every 5 --eval-games 8 --out runs/demo9x9

# GPU 9x9（约 20 分钟，torch 后端；`--device auto` 有 CUDA 就走 GPU）
# `--workers 4` 用 4 个常驻进程并行自对弈，比单进程快约 4 倍
python scripts/train.py --backend torch --device auto --size 9 --blocks 6 --channels 64 \
    --iterations 60 --games 8 --steps 40 --batch-size 128 --lr 5e-3 --lr-warmup-samples 1000 \
    --full-cap 150 --fast-cap 40 --full-fraction 0.25 --min-pass-move 40 \
    --replay-window 200000 --no-gating --eval-every 10 --eval-games 16 \
    --workers 4 --out runs/torch9x9

# 19x19 单卡（RTX 4050 6 GiB 实测可用，峰值 2.3 GB；详见下面的「19x19 训练」）
python scripts/train.py --backend torch --device auto --size 19 --blocks 6 --channels 64 \
    --iterations 100 --games 32 --steps 40 --batch-size 512 --lr 1e-3 --lr-warmup-samples 8000 \
    --full-cap 600 --fast-cap 40 --full-fraction 0.25 --min-pass-move 100 \
    --replay-window 32000 --no-gating --eval-every 25 --eval-games 16 \
    --eval-opponent random --workers 6 --out runs/torch19x19

# 论文配置（20b256c；6 GiB 上只有 368 pos/s，一局 8 分钟，跑不完一轮迭代，见 §9）
python scripts/train.py --backend torch --size 19 --blocks 20 --channels 256 \
    --iterations 100000 --games 32 --steps 500 --batch-size 256 \
    --full-cap 600 --fast-cap 100 --full-fraction 0.25 --out runs/main19x19
```

#### 19x19 训练

**显存不是瓶颈，CPU 才是。** RTX 4050 Laptop（6 GiB）上 `6b64c` 的实测：

| 指标 | 9x9 `6b64c`（150/40） | 19x19 `6b64c`（600/40） |
|---|---:|---:|
| 参数量 | 1.27 M | **10.43 M** |
| 单局（6 workers） | ~1.6 s | **~170 s** |
| 32 局 / 轮 | ~50 s | **~900 s 起**；CPU 被别的程序争用时到过 2 800 s |
| 峰值显存（`--workers 6`） | ~4.4 GB（8 worker） | **2.3 GB** |
| GPU 利用率 | — | 34~47% |

`--full-cap` 从 9x9 的 150 提到 600，原因是搜索够不够看的是 **visits / 候选数**，不是 visits 本身。
9x9 有 82 个点，150 visits 平均每点 1.8 次；19x19 有 362 个点，同样 150 次只有 0.4 次——MCTS 来不及把
候选区分开，π 基本等于网络自己的先验 P，训练就退化成"自己教自己"。600 是把覆盖率平移回 9x9 的量级。
`--fast-cap` 留在 40 不算妥协：fast 回合只负责把棋走下去，不产出样本，算力该集中在占 25% 的 full 回合上。

参数量涨了 8 倍，但**全部集中在输出头上**：policy 从 82 维变 362 维、score 从 165 bins 变 725 bins、ownership 从 81 点变 361 点，主干（6×64）一个参数没加。所以 19x19 的 1043 万并不代表更强的表示能力，真正的容量仍是那个约 100 万的主干。

几个必须跟着改的参数：

| 参数 | 9x9 常用 | 19x19 建议 | 原因 |
|---|---|---|---|
| `--full-cap` | 150 | **600** | 362 个候选，150 visits 的覆盖率只有 9x9 的 1/4 |
| `--workers` | 8 | **6** | 瓶颈在 CPU 侧的 MCTS 树操作，GPU 只用 40%，加 worker 未必更快 |
| `--batch-size` | 256 | **512** | `6×64` 的激活只有约 1.1 GB，512 仍很富裕；256×40 步对每轮 1 500~3 000 的产出只有 3~5 倍复用 |
| `--min-pass-move` | 40 | **100** | 实测一局 200~340 手，40 手就允许 pass 太早 |
| `--replay-window` | 16 000 | **32 000** | 每轮产 1 500~3 000 个样本，窗口覆盖 11~21 轮 |
| `--lr-warmup-samples` | 3 000 | **8 000** | 单样本更贵，热身期要相应拉长 |

`--eval-opponent random` 只在早期有用：30~50 轮后胜率就会饱和到 95% 以上、失去分辨力，此后应指向上一代的权重（每代只和紧邻的上一代比，胜率才会落在可分辨的 50~70% 区间）：

```bash
python scripts/train.py ... --resume runs/torch19x19_v1/model.npz \
    --eval-opponent runs/torch19x19_v1/model.npz --out runs/torch19x19_v2
```

**时间预算**：按每轮 900~1 800 s 算，100 轮要 25~50 小时。**先跑 100 轮，用第一轮的实测耗时推算后续**，
别在知道速度之前就定死 500 轮。距野狐 1d 的粗估是 30 万~60 万盘自对弈，单卡需要 2~4 个月连续运行——
这条路的价值在“每一步都可观测”，不在最终棋力。

不论棋盘尺寸，训练都会把两份权重和 `history.json`（loss / 评估曲线）写到 `--out` 目录，两份都是**覆盖式**的：

| 文件 | 写入时机 |
|---|---|
| `model.npz` | 每完成 10% 的迭代（以及最后一轮），代表「目前最新的模型」 |
| `model_best.npz` | 某一轮的 loss 创下新低时，代表「训练中 loss 最低的那一轮」 |

所以中途 Ctrl+C 也不会白跑，磁盘上至少留着最近一次 10% 节点的权重。
checkpoint 里记了 `board_size / block_count / channels / pool_channels / pool_blocks`，
所以 `play.py` / `gtp.py` 不需要再手写一遍架构；对不上时**会报错而不是静默加载**。

`--workers N` 用 N 个常驻进程并行自对弈（默认 0 = 全部在主进程里跑）。
每局之间没有依赖，所以进程级并行是安全的；每个进程持有自己的一份网络，
torch 后端还会各自建一个 CUDA context，**显存占用约为 `(N + 1) × 400 MB`**。
本机（RTX 4050 6 GiB）实测：16 局单进程约 110 s/轮，4 个 worker 降到 **23 s/轮**（4.8×）。

并行模式下 Ctrl+C 只中断主进程，spawn 出来的 worker 不会跟着退出，会继续占着显存。要彻底停下用 `taskkill /PID <主进程PID> /T /F`：`/T` 连同整棵进程树一起终止，只杀主进程会留下 8 个孤儿。另外，如果主进程正卡在控制台输出上，Ctrl+C 也会失效——Python 的键盘中断只在字节码边界检查，阻塞在系统调用里时根本没有机会处理，这种情况同样只能用 `taskkill`。

**长时间跑的训练建议脱离终端**，把输出重定向到文件：

```powershell
# 目录要先建好：重定向是 shell 在启动 Python 之前处理的，
# 目录不存在的话 python 根本起不来（train.py 里的 makedirs 来不及执行）
New-Item -ItemType Directory -Force runs/torch9x9_v4 | Out-Null

python -u scripts/train.py --backend torch --device auto --size 9 --blocks 6 --channels 64 `
    --iterations 500 --games 32 --steps 40 --batch-size 256 --lr 1e-3 --lr-warmup-samples 3000 `
    --full-cap 150 --fast-cap 40 --full-fraction 0.25 --min-pass-move 40 `
    --replay-window 16000 --no-gating --eval-every 50 --eval-games 64 `
    --workers 8 --out runs/torch9x9_v4 --resume runs/torch9x9_v3/model.npz `
    --eval-opponent runs/torch9x9_v3/model.npz > runs/torch9x9_v4\train.log 2>&1
```

`-u` 关掉输出缓冲，`>` 让 stdout 变成文件而不是控制台。这样训练进程和终端彻底解耦：窗口被选中、被暂停、甚至被关掉都不影响训练，Ctrl+C 也不再是唯一的停止手段。看进度另开一个窗口：

```powershell
Get-Content runs/torch9x9_v4\train.log -Wait
```

查看用的窗口可以随便点，但**不要点训练所在的窗口**——被选中的控制台会暂停输出，又会回到需要 `taskkill` 的状态。

`--eval-opponent` 指定评估环节的对手，默认 `random`：

| 取值 | 行为 |
|---|---|
| `random` | 均匀随机合法落子，`--eval-games` 局黑白各半 |
| `runs/xxx/model.npz` | 与指定 checkpoint 对局；权重启动时加载一次，此后**冻结不变** |

指定 checkpoint 时双方用相同的搜索量（`max(8, --fast-cap)`），所以胜率反映的是权重差异而不是搜索预算差异。对手网络是独立的一份，额外占约 400 MB 显存，每轮 eval 的耗时也大约翻倍。路径在启动时就会加载并校验，棋盘尺寸对不上会立即报错。

随机对手在胜率超过 ~80% 之后就没有分辨力了（64 局的标准误是 ±5%），而这个区间正是训练后期待的地方。让新一轮打上一轮的成绩，才能看出是否真的在进步：

```bash
python scripts/train.py ... \
    --eval-opponent runs/torch9x9_v2/model.npz --out runs/torch9x9_v3
```

日志里会写成 `eval @iter 99: win rate vs model.npz 63.0%`，稳定高于 50% 才说明新模型更强。

#### 云端训练（Linux）

租用 GPU 时最要紧的不是参数，而是**让训练活过 SSH 断线**。直接在登录 shell 里前台启动，
网络一抖就全没了：

- SSH 断开时终端给前台进程组发 `SIGHUP`，python 直接退出
- 如果还套了 `| tee`，tee 写终端失败会拿到 `SIGPIPE`，python 连 checkpoint 都来不及存

`nohup` 能挡住 `SIGHUP`，但拿不回实时输出、也不能中途 Ctrl+C。**tmux 两者兼顾**：

```bash
# 一次性离开终端也能跑；实例只要处于「运行中」就一直计费运行
tmux new -s train

# 在会话里启动，tee 让屏幕和文件同时有输出
python -u scripts/train.py --backend torch --device auto --size 19 --blocks 6 --channels 64 \
    --iterations 100 --games 32 --steps 40 --batch-size 512 --lr 1e-3 --lr-warmup-samples 8000 \
    --full-cap 600 --fast-cap 40 --full-fraction 0.25 --min-pass-move 100 \
    --replay-window 32000 --no-gating --eval-every 25 --eval-games 16 \
    --workers 6 --out runs/19x19_v1 2>&1 | tee runs/19x19_v1/train.log
```

看到 `self-play pool: 6 worker processes` 之后按 `Ctrl+B`、松开再按 `D` 脱离。回来用 `tmux attach -t train`，
想停就在会话里 `Ctrl+C`——走的和本机一样，会正常写 checkpoint。`nohup` 也可以，但需要
`kill -INT <pid>` 才能触发保存（`kill -9` 会跳过 checkpoint 写盘）。

跑之前确认两件事：

| 项 | 说明 |
|---|---|
| 目录存在 | `mkdir -p runs/19x19_v1`；`tee` 和 `>` 都是 shell 在启动 python **之前**打开的，建目录晚了会直接失败 |
| 盘位 | AutoDL 的系统盘只给 30 G，数据盘 `/root/autodl-tmp` 更大更快；代码放系统盘（能随镜像保存），`runs/` 放数据盘 |

Pascal 架构的卡（1080 Ti / 1080 / Titan X）**只能用 FP32**：它们的 FP16 是 1/64 速率。本项目没有
`autocast`，全程 FP32，所以不受影响，但要锁定 `torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118`——
更新的版本可能不再提供 `sm_61` 的预编译 wheel。

**续训必须让起点和对手同代**，否则评估没有意义（对 `random` 会在 50 轮内饱和到 95% 以上）。
在上面那条命令上加两个参数，`--lr-warmup-samples` 同步降到 3000（训练过的网络不需要长爬坡）：

```bash
--resume runs/19x19_v1/model.npz --eval-opponent runs/19x19_v1/model.npz \
    --lr-warmup-samples 3000 --out runs/19x19_v2
```

第一轮 eval 因此应该接近 50%，之后每一点涨幅都是真实的。注意 `--resume` 只恢复权重，
**不恢复 replay buffer**，所以续训头 20 轮（32000 ÷ 每轮样本）loss 会抖——那是缓冲区从空到满的过渡。

云端实例被释放后磁盘不保留，按小时计费时记得把权重拉回本地：

```bash
scp user@host:~/code/mini-katago-pytorch/runs/19x19_v1/model.npz ./backup_19x19_v1.npz
```

`model.npz` 每 10% 迭代写一次，所以即使漏备份，最多丢 10 轮。**SSH 断线不影响训练，主动「关机」会**：
nohup 和 tmux 都挡不住实例停机，只能用余额和备份兜底。

### 对弈

```bash
# 图形界面（鼠标落子，Tkinter，无需额外依赖）
python scripts/gui.py --checkpoint runs/demo9x9/model.npz --visits 200

# 打开就用分析模式（打谱 + 实时分析）
python scripts/gui.py --checkpoint runs/demo9x9/model.npz --mode analyze --visits 2000

# 人机对弈（终端棋盘）
python scripts/play.py --checkpoint runs/demo9x9/model.npz --size 9 --visits 200

# 引擎自战，并存成 SGF
python scripts/play.py --checkpoint runs/demo9x9/model.npz --self --save-sgf game.sgf

# 开局随机 4 手 + 在访问数相近的候选里采样，每次都是新棋
python scripts/play.py --checkpoint runs/demo9x9/model.npz --self \
    --opening-moves 4 --temperature 0.5 --min-visit-ratio 0.3 --top-k 3
```

`--temperature 0`（默认）永远下访问数最多的一选，而且搜索不加根噪声，所以
同一份权重、同一个空盘开局，**每次产出的棋谱完全一样**。想让对局有变化：

| 参数 | 作用 |
|---|---|
| `--opening-moves N` | 开局先均匀随机走 N 手，换一条局面序列 |
| `--temperature T` | `0` = 永远一选；`0.5` 偏向一选、二选偶尔出现；`1.0` 严格按访问数比例采样；`2.0` 接近均匀（棋力会明显下降） |
| `--min-visit-ratio R` | 只保留访问数 ≥ 第一名 `R` 倍的候选，挡住只搜了一两次的废棋 |
| `--top-k K` | 候选硬上限，只在前 K 名里选 |
| `--seed S` | 固定随机种子以复现某一盘（默认每次新种子） |

采样权重是 `N ** (1 / T)`：`T=1` 时一选 42%、二选 42% 就是一个真正的 50/50；
`T=0.5` 时权重变成 `N²`，一选压倒性领先。输出里带 `(sampled)` 的着手就是采样所得。

GUI 有两种模式，用「模式」开关切换，切换时保留当前棋局：

- **对弈**：鼠标落子、悔棋、Pass、认输、切换执子、调整搜索量，以及基于
  ownership 头的实时形势图。
- **分析**：双方自由落子打谱，引擎报告一选到十选的选点、访问量、胜率与目差；
  「分析量」可填任意 visit 上限，搜索过程中候选排名和数值边搜边刷新（棋盘上
  以蓝色数字标出前十），到达上限后自动停住并保留结果。按「停止」可提前结束，
  落子则立即按当前上限重新分析。

引擎在后台线程搜索，窗口不会卡住；`--backend auto` 会跟随 checkpoint 记录
的训练后端，因此 NumPy 训练的权重不会被误交给 PyTorch 加载。

窗口可以自由拖拽缩放，棋盘会跟着取窗口里最大的正方形，拖到停下才重绘；
`F11` 在窗口化和全屏之间切换，`Esc` 退出全屏。侧栏宽度不变，窗口最窄时
会保证分析面板（十个候选 + 信息区）完整显示。

侧栏是一整块白色卡片配一条发丝边，浮在浅灰底上：标题旁的小标签标出后端，
设置项按左侧灰色列名对齐，主操作（新对局 / 重新分析）是蓝色实心按钮，其余
是浅灰描边按钮，悔棋清谱这类不可逆操作单独用红色；最下面的状态卡带一条
竖色条，思考中变蓝、终局变红。候选表隔行浅底，前三名加粗。

外观在侧栏的「棋盘」「棋子」两个下拉里切换，切换只重绘画布，棋局和正在跑的
搜索都不受影响；也可以在启动时指定：

```bash
python scripts/gui.py --checkpoint runs/demo9x9/model.npz \
    --board-style space --stone-style gem
```

- 棋盘：`wood` 暖色木纹、`stone` 灰色石材带斑点与石纹、`space` 深蓝星野
  （星云 + 两层星点）。纹理由固定种子生成，缩放窗口后图案保持一致。
- 棋子：`classic` 黑白、`gem` 玛瑙（红棕纹路）与翡翠（淡绿）。棋子用
  「边缘 → 暗环 → 主体 → 左上高光 → 右下反光」几层同心圆叠出球面感，
  下面再压两层投影，投影颜色跟随棋盘主题。

落子声是 `katago/assets/stone_click.wav` 里的实录声（单声道 16-bit PCM，
约 0.2 秒）。侧栏「落子声」勾选可随时开关，`--no-sound` 则启动即为静音，
Pass 不出声。Windows 用 `winsound` 异步播放，macOS 调系统自带的 `afplay`，
其它平台退化为窗口提示音。录音缺失时会退回标准库实时合成的木盘「嗒」声，
所以文件丢了也不影响落子。

### 接图形界面（GTP）

```bash
python scripts/gtp.py --checkpoint runs/demo9x9/model.npz --size 9 --visits 400
```

支持 `boardsize / komi / clear_board / play / genmove / undo / showboard /
kata-analyze / quit`，坐标遵循 GTP 规则（列字母跳过 `I`）。

### 基准测试

```bash
python scripts/benchmark.py --backend numpy --size 9 --blocks 3 --channels 24
```

### 跑测试

```bash
python tests/run_tests.py          # 不依赖 pytest
pytest -q                          # 装了 pytest 也可以
```

---

## 2. 项目结构

```
katago/
├── game/                 围棋规则与输入特征
│   ├── rules.py          规则配置（棋盘尺寸 / 贴目 / 劫规则 / 禁止自杀）
│   ├── board.py          Position：提子、劫、superko、Tromp-Taylor 数子、
│   │                     Benson 活棋、归属（ownership）
│   ├── ladder.py         有界征子（ladder）读取器
│   └── features.py       18 个空间平面 + 10 个全局特征
├── nn/                   神经网络
│   ├── config.py         NetConfig / MCTSConfig / PlayoutCapConfig / LossConfig / TrainConfig
│   ├── base.py           BaseNet 接口 + EvalResult
│   ├── model.py          ★ PyTorch 实现（主）
│   ├── numpy_net.py      NumPy 参考实现（无 BN）
│   └── autograd.py       小型反向模式自动微分引擎
├── mcts/
│   └── search.py         ★ PUCT / FPU / Dirichlet 噪声 / forced playouts /
│                         policy target pruning / 批量虚拟损失
├── selfplay/
│   ├── worker.py         自对弈一局，产出 policy / value / ownership / score / aux 目标
│   ├── pool.py           多进程自对弈池（`--workers`）：常驻 worker + 按轮广播权重
│   ├── replay.py         滑动窗口回放 + 8 重对称增广
│   ├── gating.py         门控测试（候选网络 vs 当前网络）
│   └── trainer.py        训练闭环：SWA、门控、渐进加宽、学习率调度
├── inference/
│   ├── engine.py         analyze() / genmove() / AnalysisSession 增量分析
│   └── gtp.py            GTP 前端
├── utils/
│   ├── symmetry.py       8 种二面体变换
│   ├── sgf.py            SGF 导出
│   └── sound.py          落子声：优先播 assets 里的录音，缺失则实时合成
└── assets/
    └── stone_click.wav   实录落子声（单声道 16-bit PCM，约 0.2 秒）

scripts/   train.py  play.py  gtp.py  gui.py  benchmark.py
tests/     run_tests.py + 10 个测试模块（118 项）
```

---

## 3. 系统架构

```
                     ┌─────────────────────────────────────────────┐
                     │           神经网络 f_θ (PyTorch)             │
                     │                                             │
  18 平面 ┐          │  5x5 conv ─┐                                │
  10 全局 ┴──────────►│            ├─► n × Pre-act ResBlock         │
                     │  FC → bias ┘   (其中 2~3 个换成 Global       │
                     │                 Pooling Block)               │
                     │                     │                       │
                     │        ┌────────────┼────────────┐          │
                     │        ▼            ▼            ▼          │
                     │   Policy head   Value head   Aux heads      │
                     │   (b²+1)        (win/loss)   ownership      │
                     │                              score pdf/cdf  │
                     │                              opp-policy     │
                     └───────────┬─────────────────────────────────┘
                                 │  P, V, ownership, score
                                 ▼
        ┌────────────────────────────────────────────────┐
        │  MCTS（PUCT + FPU + Dirichlet + forced playouts）│
        │  → π（经 policy target pruning）                │
        └───────────────┬────────────────────────────────┘
                        │ 落子（温度 τ）
                        ▼
        ┌────────────────────────────────────────────────┐
        │  自对弈：playout cap randomization              │
        │  p=25% 回合做全搜索 → 只有这些回合被记录       │
        └───────────────┬────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────┐
        │  滑动窗口回放 + 8 重对称增广                    │
        └───────────────┬────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────┐
        │  SGD(momentum) → SWA(4 快照, decay 0.75)        │
        │     → 门控测试(≥50%) → 成为自对弈网络 θ*        │
        └────────────────────────────────────────────────┘
```

**数据流（一次自对弈的一回合）**

| 阶段 | 输入 | 输出 |
|---|---|---|
| 编码 | `Position` | `(18,S,S)` 平面 + `(10,)` 全局向量 |
| 网络 | 编码 | policy 先验 P、value V、ownership、score 分布 |
| 搜索 | P, V + 规则 | 访问计数 N（→ π） |
| 记录 | π, 最终胜负/比分/归属 | 训练样本 |

---

## 4. 与论文的逐项对应

### 4.1 网络结构（Wu 2020, Appendix A）

| 论文条目 | 实现位置 | 状态 |
|---|---|---|
| 18 个空间二值平面 | `game/features.py` | ✅ 全部 18 个 |
| 10 个全局实数特征 | `game/features.py` | ✅ |
| 5×5 首卷积 + 全局向量 FC 作通道偏置 | `nn/model.py: KataGoNet.trunk` | ✅ |
| Pre-activation 残差块 (BN→ReLU→3×3→BN→ReLU→3×3→skip) | `nn/model.py: ResBlock` | ✅ |
| **Global Pooling Bias 结构**（Fig. 2） | `nn/model.py: GlobalPoolingBias` | ✅ |
| 全局池化层：mean / mean×尺度 / max | `nn/model.py: GlobalPoolingLayer` | ✅ |
| value head 用二次尺度替换 max | `GlobalPoolingLayer(value_mode=True)` | ✅ |
| 2~3 个残差块使用全局池化（均匀间隔） | `NetConfig.pool_block_indices` | ✅ |
| Policy head：1×1 conv → BN → ReLU → GPB → FC | ✅ | ✅ |
| Ownership / Score pdf / Score cdf / Aux-policy 头 | ✅ | ✅ |
| 渐进加宽 (6,96)→(10,128)→(15,192)→(20,256) | `selfplay/trainer.py: _start_widening` | ✅ 已实现；需要在 `TrainConfig.widening` 里显式配置，CLI 没暴露 |

**18 个平面**（Table 3）：

```
0      在盘内
1-2    己方 / 对方棋子
3-5    气数为 1 / 2 / 3 的棋子
6      因劫 / superko 而不能落子
7-11   最近 5 手（最新的在前，pass 留空）
12-14  0/1/2 手前落下的棋子正处于征子中
15     落在此处可以征掉对方
16-17  Benson 活棋区域（己方 / 对方）
```

**10 个全局特征**（Table 4）：最近 5 手是否 pass、`komi/15`、劫规则 one-hot(2)、
是否允许自杀、`komi + 棋盘尺寸奇偶`。

### 4.2 搜索（Section 2 / 3.1 / 3.2）

| 技巧 | 实现 | 默认值 |
|---|---|---|
| PUCT：`V + c·P·√ΣN/(1+N)` | `MCTS._puct_scores` | `c_puct = 1.1` |
| FPU：`V(c) = V(parent) − c_fpu√P_explored` | `MCTS._puct_scores` | `c_fpu = 0.2` |
| 根节点 Dirichlet 噪声 | `MCTS.set_root` | `P=0.75P+0.25Dir(α)`，`α=0.03·19²/N` |
| 根节点 softmax 温度 1.03（SAI 技巧） | `MCTS.set_root` | `root_policy_temp=1.03` |
| **Forced playouts** | `MCTS._compute_forced_targets` / `_select_child` | `n_forced=√(k·P·ΣN)`，`k=2` |
| **Policy target pruning** | `MCTS._prune_counts` / `policy_target` | 开 |
| **Playout cap randomization** | `selfplay/worker.py` | `p=0.25`，`(N,n)=(600,100)` |
| 认输 | `worker.py` | 根节点 Q < −0.95 且访问数 ≥100 |

表中的技巧都来自论文（表中的数值是论文主运行的常数；CLI 默认是 `--full-cap 80 --fast-cap 20`）。
另外还有一项**工程实现**，不属于论文的方法贡献：
批量叶节点评估（virtual loss）在 Wu 2020 正文里从未出现，只在 §5.1 的比赛设置里
提了一句 KataGo 引擎用了 “16 threads, max batch size 16”，既没有算法描述也没有
消融实验。本项目用 `MCTS.run(batch_size=...)` 实现，CLI 是 `--leaf-batch`（默认 16）。

### 4.3 训练（Section 2 / 4.1）

| 技巧 | 实现 | 默认值 |
|---|---|---|
| 损失：`c_g·CE(value) + CE(policy) + c_L2·‖θ‖²` | `nn/*.train_batch` | `c_g=1.5`，`c_L2=3e-5` |
| 辅助 policy 目标（对手应手） | `worker.GameRecord.samples` | `w_opp=0.15` |
| Ownership 目标（0/0.5/1） | 同上 | 见 §6 说明 |
| Score pdf + cdf | 同上 | `w=0.02` 各 |
| SGD momentum | `autograd.SGD` / `torch.optim.SGD` | `0.9` |
| 学习率 + warmup + 衰减 | `TrainConfig.learning_rate` | 论文 `6e-5` 是**每样本** lr @ batch 256；CLI 的 `--lr` 是作用在 batch-mean 梯度上的步长，对应 `6e-5 × 256 = 0.0154` |
| 滑动窗口回放 | `ReplayBuffer` | `250k → 22M` |
| SWA（4 快照，EMA decay 0.75） | `Trainer._swa_candidate` | ✅ |
| 门控测试（100/200） | `gating.gating_test` | ✅ |
| 8 重对称增广 | `utils/symmetry.py` | ✅ |
| 规则/棋盘尺寸随机化 | `Rules` + `Rules.successor` | ✅（可选） |

---

## 5. 关键实现细节

### 5.1 Forced playouts 与 policy target pruning

这是 KataGo 最核心、也最容易实现错的两个技巧。

**问题**：AlphaZero 的 policy 目标是根节点最终的 playout 分布，而这个分布里混进了
探索噪声（根节点 Dirichlet 噪声）和 PUCT 自身的探索偏差，于是网络被训练去高估坏棋。

**Forced playouts**（Section 3.2）：对每个在根节点获得过 playout 的子节点 `c`，保证它至少得到

```
n_forced(c) = ( k · P(c) · Σ N(c') )^(1/2)
```

个 playout。实现方式是：当某个根子节点的访问数少于这个目标时，把它的选择 "urgency"
设成无穷大。

- 指数 `1/2 < 1` 保证 forced playouts 随搜索规模增长但在坏棋上的**比例渐近趋于 0**；
- 之所以需要它，是因为作者观察到：某步棋哪怕被 Dirichlet 噪声选中了、且本身是好棋，
  如果**首次评估为负**，就再也不会被搜索到，好棋被永久埋没。

**Policy target pruning**：forced playouts 灌进来的绝大多数是坏棋，直接拿分布当目标
等于给网络喂噪声。所以再把它减出去：找到访问最多的子节点 `c*`，对每个其他子节点，
在不使 `PUCT(c) ≥ PUCT(c*)` 的前提下，最多减去 `n_forced(c)` 个 playout；
被减到只剩 1 个 playout 的子节点直接剔除。

**关键洞察**（论文自己的说法）：这实现了 **policy 目标与 MCTS 探索动态的解耦** ——
"没有理由认为，MCTS 中最优的 playout 离散程度，对 policy 目标的长期收敛也是最优点"。

代码上，`MCTS._prune_counts` 是纯函数（不修改树），便于单独做确定性单测
（`tests/test_mcts.py::test_prune_counts_is_deterministic_and_budget_limited`
用手工构造的树验证了"好棋一个不减、坏棋减满预算、中间棋减到刚好还会被选中"。

### 5.2 Playout cap randomization（Section 3.1）

**问题**：value 目标极度稀缺（一局只有一个带噪的二值结果），policy 目标则相反（需要足够深的
搜索才能让 π 偏离 P）。固定 N 无法同时满足二者。

**解法**：`p` 比例的回合做全搜索（cap `N`），其余做快搜索（cap `n < N`），
**只记录全搜索回合**；快搜索回合关闭 Dirichlet 噪声以最大化棋力。

主运行取 `p=0.25`、`(N,n)=(600,100)`，两天后退火到 `(1000,200)`。
`PlayoutCapConfig.caps(samples_seen)` 实现了同样的退火。

### 5.3 Global pooling

纯卷积感受野有限，而围棋有明显的非局部交互（最典型是**劫**）。全局池化把
`(mean, mean×尺度, max)` 三个统计量投影成另一个通道组的逐通道偏置，让卷积层能
"conditioned on global context"。

尺度因子 `(b − b_avg)/10`（value head 用 `((b−b_avg)² − σ²)/100`）是为了让**同一套权重
跨棋盘尺寸通用**：比分差随棋盘宽度**二次**增长，所以 value head 必须用二次尺度。

### 5.4 辅助目标为什么有效

论文给的直觉值得复述：设想从一局**主要输在某个局部误判**的对局里更新。只有最终二值结果时，
网络只能"猜"是哪个方面导致了失败；而有了 ownership 目标，网络会收到直接反馈——
**是盘面哪一块区域预测错了**，大误差和大梯度**集中在该区域**，因此需要更少的样本就能完成
正确的 credit assignment。

推广开来就是论文提炼的元学习启发式：**预测目标的分量（sub-components）比预测目标本身
提供了更细粒度、更局部化的监督信号**。

---

## 6. 与论文的偏差（必读）

诚实列出所有与论文不完全一致的地方，以及为什么。

| # | 偏差 | 原因 / 影响 |
|---|---|---|
| 1 | **torch 路径只在中小规模上验证过** | torch 2.6.0+cu118 / RTX 4050 6 GiB 上已验证：`tests/test_torch_model.py` 7 项全过，9×9 与 19×19 的 `6b64c` 都能完整训练（19×19 在 `--full-cap 600 --fast-cap 40` 下单局约 170 s @ 6 workers，峰值 2.3 GiB）。论文级配置（`20×256`、全手 600 playouts）没跑过：该网络只有 368 pos/s，一局要 8 分钟，本机跑不完一轮迭代 |
| 2 | **NumPy 后端没有 BatchNorm** | 自动微分的范围控制。NumPy 后端与 torch 拓扑一致（pre-act 残差 + 全局池化偏置 + 多输出头），仅去掉 BN；它只用于验证流水线，不追求棋力 |
| 3 | **ownership 损失的权重换算** | 论文对 `b²` 个点**求和**并乘 `1.5/b²`；本实现对点**取平均**，因此 `1.5/b²` 的分母与之抵消，实际权重为 1.5。见 `LossConfig.w_ownership` |
| 4 | **score cdf 损失按论文公式无法成立** | 论文写作 `Σ_x (Σ_{y<x} p_s(y)·p̂_s(y))²`，这在 `p_s` 是 one-hot 时退化为"真实分数之上的 bin 数 × p̂(x*)²"，不是合理的损失。本实现采用显然的意图：**预测 CDF 与真实 CDF 的平方误差** |
| 5 | **ladder 读取器是有界的近似** | 实现了"连续打吃 + 逃子检测 + 断梯检测"的核心循环，深度上限 `2·b+4`；不确定时返回 False（宁可漏报，不可误报）。不是 KataGo 那套完整梯子求解器 |
| 6 | **不做树复用** | 每手都重建搜索树。正确但不经济；真实 KataGo 会把上一手的子树接过来 |
| 7 | **门控在演示中关闭** | 论文的门控按 ~1M 样本的节奏触发；演示总共只有几千样本，若开门控则自对弈网络永不更新。`--no-gating` 让自对弈网络每次迭代直接跟随训练权重。真实规模请勿关闭 |
| 8 | **演示用了远超论文的学习率** | 论文 `6e-5` 每样本 lr 是按百万级样本、几十万步设计的；CPU 演示只有约 1500 步，因此用 `5e-3`。这不是"论文参数"，是为了在几分钟内看到信号 |
| 9 | **小增益初始化输出头** | 论文未明确写出，但 AlphaGo Zero / KataGo 都这样做：随机初始化的 policy 头会**异常自信**，导致自我对弈第一步就变成确定性走法，搜索分布退化成 one-hot，训练完全学不动（这是我在本项目中实测到的真实故障，见 §7.3）。因此 policy/aux/ownership/score/win 五个输出层用 `std=0.01` 初始化 |
| 10 | **自对弈前 N 手禁用 pass** | 演示规模下网络会收敛到"立刻 pass"的退化均衡：Tromp-Taylor 数子在**近乎空盘**上会把整盘判给只有一颗子的一方，于是"第 2 手结束对局"对 value 目标来说**看起来是大胜**。真实规模下搜索会自己发现"对手会继续下"，CPU 演示到不了那里。`SelfPlayConfig.min_pass_move` 只在自对弈时生效，**推理不受影响**。详见 §7.3 |
| 11 | **自对弈并行度由 `--workers` 控制** | 论文用 16~24 块 V100 并行生成（Appendix C）。本实现提供 `--workers N`：N 个常驻进程各持一份网络、独立下棋，权重按迭代广播，默认 0（单进程）。实测 9×9 / 6b64c / 16 局：单进程约 110 s/轮，4 worker 约 23 s/轮（4.8×）。瓶颈不在 GPU：`legal_mask` 和 `encode_position` 占自对弈时间 ~60%，GPU 前向只占 ~10%，所以只能靠进程并行（线程受 GIL 限制无效）。每个 worker 会建自己的 CUDA context（~300–500 MB 显存） |
| 12 | **棋盘尺寸随机化未在演示中启用** | 代码支持（`Rules.successor`、可变尺寸网络、全局池化的尺寸尺度），但演示固定 9×9 |

---

## 7. 测试与验证

### 7.1 测试覆盖

```
tests/test_game.py       35 项  规则引擎：提子/自杀/三种劫/Benson/Tromp-Taylor/征子/18 平面特征
tests/test_autograd.py   13 项  有限差分梯度检查（含 conv2d、max pooling、CDF 损失）
tests/test_numpy_net.py   5 项  前向形状、概率归一、状态字典、固定批次收敛、真实局面策略拟合
tests/test_mcts.py       15 项  PUCT/FPU/forced playouts/pruning/温度采样/批量一致性/PV 合法性
tests/test_selfplay.py   17 项  自对弈记录、cap randomization、价值目标、aux 目标、回放、增广、GTP
tests/test_trainer.py     6 项  学习率调度、窗口增长、SWA 权重、存档往返、端到端 loss 下降
tests/test_checkpoint.py  4 项  存档结构、从权重反推架构、架构不匹配必须报错
tests/test_extras.py      5 项  SGF 导出、渐进加宽提升为更大的网络
tests/test_torch_model.py 7 项  PyTorch 专属（装了 torch 才会真跑，否则 SKIP）
```

一条命令（约 1.5 分钟，因为里面真的在跑自我对弈）：

```bash
python tests/run_tests.py
```

### 7.2 三个值得单独说的验证

**(a) 自动微分逐算子梯度检查。** `tests/test_autograd.py` 对每个算子做中心差分比对，
`conv2d`（pad 0/1、stride 1）、`max_hw`、`concat/narrow`、softmax-CE、sigmoid-BCE、
CDF 损失都通过。这是 NumPy 后端能作为"验证工具"的前提。

**(b) 目标剪枝的确定性单测。** 见 §5.1。手工构造 `prior=[0.6,0.3,0.1]`、
`N=[100,10,3]`、子节点均值 `[0.1,0.8,0.9]` 的树，断言：
好棋 100 个 playout 一个不减；坏棋减满 forced 预算（3→1）；中间棋减到刚好还会被选中（10→5）。

**(c) GTP 坐标往返。** 列字母跳过 `I`，`A9`/`E5`/`J9` 等边界都验证过。

### 7.3 调试中真实踩到的坑（保留在此，因为它们解释了代码里的设计）

1. **随机初始化的输出头导致自我对弈退化。** 未做小增益初始化时，随机网络的 policy 几乎
   是 one-hot，于是 MCTS 的访问分布也几乎 one-hot，训练信号退化为"预测均匀分布"。
   实测：loss 从 11.05 掉到 7.5 后完全不动，policy loss 停在 `ln(82)=4.41`、
   value loss 停在 `ln(2)=0.693`——典型的"全部预测均匀"。
   加了 0.01 增益初始化后，同样 `cap=100` 下策略目标从"1.2 个非零项"变成"11.3 个非零项、
   熵 2.39"，训练才开始动。
2. **剪枝在小预算下会把目标清空。** 论文的"剔除只剩 1 个 playout 的子节点"在
   几十个 playout 的预算下会清空整个分布。`policy_target` 因此加了回退链：
   剪枝分布 → 未剪枝分布 → 先验。
3. **Benson 算法少了 "vital region" 条件。** 第一版用"区域只与一条棋链相邻"，
   结果空盘上**孤零零一颗子也被判成活棋**。正确判据是：区域必须 *vital* 于某条链
   （区域内每个空点都是该链的气），且每条链至少 vital 于两个区域。
4. **自对弈会收敛到"立刻 pass"的退化均衡。** 这是本项目最有意思的一个坑。第一次演示
   跑完，320 局自对弈里平均每局只有 **17 手**，赢率对随机玩家停在 12%~37%。
   查下去发现对局是以"连续两次 pass"结束的，且第 1 手之后立刻 pass。

   原因不在搜索，而在**数子规则**：Tromp-Taylor 区域计分下，只要盘上只有黑棋，
   "只被一种颜色包围的空区"全都算这方的地——空盘上摆一颗子，黑就"拥有"整盘 81 点。
   于是"我下一子，然后双方 pass"对 value 目标来说是一局**大胜**，网络很快学会了
   这个局部最优。真实规模的 KataGo 不会卡在这里，因为对手会继续下、把地围走；
   但几百局的预算下，网络还没机会学到这一点。

   处理办法是 `SelfPlayConfig.min_pass_move`：**自对弈前 N 手不允许 pass**
   （同时把这些回合的 policy 目标里的 pass 质量零）。这是训练稳定性措施，
   推理时 pass 依然合法。这个坑也解释了为什么 AlphaGo Zero 需要 490 万局——
   很多"显然"的道理必须靠足够多的对局才能被搜索发现。

---

## 8. 演示结果

配置（`runs/demo9x9/`）：9×9 棋盘、`komi 7.5`、2 个残差块、16 通道（15.1 万参数）、
每手 48(全)/12(快) playouts、`p=0.25`、30 次迭代 × 4 局自对弈、NumPy 后端、单 CPU 核。
**总耗时 32.5 分钟**，生成 **3,165** 个训练局面、训练 **48,000** 个样本（120 局自对弈）。

### 8.1 训练曲线

| iter | 总 loss | policy | value | ownership | score pdf | 对随机玩家胜率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.198 | 4.296 | 0.589 | 0.692 | 5.012 | — |
| 4 | 6.801 | 3.872 | 0.580 | 0.596 | 4.033 | 18.8% |
| 9 | 6.609 | 3.821 | 0.477 | 0.613 | 4.297 | 62.5% |
| 14 | 6.570 | 3.882 | 0.517 | 0.538 | 3.901 | 50.0% |
| 19 | 6.405 | 3.901 | 0.486 | 0.582 | 4.212 | 62.5% |
| 24 | 6.565 | 3.767 | 0.524 | 0.627 | 4.408 | 62.5% |
| 29 | **6.010** | **3.329** | 0.505 | **0.613** | **4.121** | **75.0%** |

参照量：9×9 上有 82 个合法动作，**均匀分布的 policy 交叉熵 = ln 82 = 4.407**；
165 个比分 bin 上均匀分布的 CE = 5.11；ownership 上恒输出 0.5 的 BCE = 0.693。
所以 policy 从 4.296 降到 3.329 意味着网络确实学到了东西，而不是停在最大熵。

### 8.2 最终棋力

用训练好的权重跑 8 局对随机玩家（每手 100 visits，黑白各 4 局）：

```
游戏长度   109, 97, 117, 118, 106, 87, 99, 95 手
比分       +1.5, -7.5, +21.5, -33.5, +9.5, -10.5, +3.5, -8.5
结果       8 胜 0 负
```

终局棋盘示例（模型执黑，`X` = 黑，`O` = 白）：

```
   A B C D E F G H J
 9 O O X X X X . X O 9
 8 O O X O O O O X O 8
 7 O O X O O O O O O 7
 6 O O O O O . O O O 6
 5 O O X O O O O O . 5
 4 X X X O O O X O O 4
 3 X X X X X . X X X 3
 2 X X X X X X X X X 2
 1 X X . X X X X X X 1
   A B C D E F G H J
```

对局长度 87~118 手、双方都有正常的模样与地盘——这正是"立刻 pass"退化解被修复之后
应有的形态（对比 §7.3 第 4 条）。

### 8.3 怎么解读这个结果

**它是一个"流水线跑通"的证据，不是棋力复现。** KataGo 主运行是 1.4 GPU-years
（约 1.2 万 GPU-小时），本演示是 0.5 CPU-hours，相差约 5 个数量级。论文的结论恰恰是
"AlphaZero 的方法离自我对弈的上限还很远"，那需要论文级的算力才能看到。

能验证的是：规则引擎、18 平面特征、全局池化网络、KataGo 搜索的三个专属技巧、
多目标损失、回放与增广、训练闭环——这些**逐项被单元测试和这条端到端曲线共同覆盖**。

在装了 PyTorch 的机器上，把 `--backend torch`、`--size 19 --blocks 20 --channels 256`、
`--games 32 --steps 500 --full-cap 600 --fast-cap 100` 接上去，就是论文的配置。

---

## 9. 性能

NumPy 参考后端在 9×9、3 块、24 通道下的实测（`scripts/benchmark.py`）：

| 项目 | 数值 |
|---|---|
| `encode_position`（缓存已热） | 0.15 ms |
| 前向 `batch=1` | 2.96 ms（≈338 pos/s） |
| 前向 `batch=8` | 8.70 ms（≈919 pos/s） |
| 搜索（64 playouts，叶批量 8） | 2.10 ms/position（≈476 pos/s） |

PyTorch 后端在 RTX 4050（6 GiB, 20 SMs）上的实测：

| 网络 | 推理 @batch8 | 训练一步 | 峰值显存 |
|---|---:|---:|---:|
| 9×9 `4b32c` | 2717 pos/s | 未测 | 0.05 GiB |
| 9×9 `6b64c` | 2195 pos/s | 30 ms @128 | 0.18 GiB |
| 13×13 `6b64c` | 2341 pos/s | 41 ms @128 | 0.27 GiB |
| 19×19 `6b64c` | 未单独测 | 未测 | 2.3 GiB（`--workers 6` 自对弈） |
| 19×19 `10b128c` | 1537 pos/s | 266 ms @128 | 1.51 GiB |
| 19×19 `20b256c`（论文） | 368 pos/s | 615 ms @64 | 2.96 GiB |

**自对弈的瓶颈不在 GPU。** 一次 9×9 / `6b64c` / cap 100/25（25% 全搜索）的自对弈里，
`legal_mask` 与 `encode_position` 合计约占 60%，网络前向只占 10%，其余是搜索本身
的 Python 开销。19×19 上结论一样：`--workers 6` 跑自对弈时 GPU 利用率只有 34~47%、
功耗 11 W（上限 40 W），显存也只用掉 2.3 GB——多开 worker 收益递减，真正的瓶颈在 CPU。
按 profile 结果做了四项优化：

1. `Position` 构造时只算一次棋盘 hash（原来 `_hash` 和 `_hash_stack_board` 各算一遍）；
2. `is_legal` 在**没有提子**时直接返回合法，跳过 superko 的全盘 hash —— 不提子的
   落子只增加棋子，不可能重现历史局面；
3. `predict` 把五个输出头的激活在 GPU 上算完后**一次**搬回主机，而不是五次
   `.cpu()`（每次都强制一次 stream 同步）；
4. 叶节点批量从 8 提到 16（`--leaf-batch`）；32 以上吞吐继续上升，但虚拟损失把
   对局拉长到 120+ 手，采样质量下降，不可取。

9×9 / `6b64c` / cap 100/25（25% 全搜索）/ `--steps 50 --batch-size 128` 的端到端对比：
**84.7 s → 46.8 s**（3 轮迭代，1.81×）。想再快就得重写规则引擎（整盘一次连通
分量分析代替逐点 flood fill），那是另一个量级的改动。

---

## 参考文献

- D. Silver et al., *Mastering the game of Go without human knowledge*, Nature 550, 354–359 (2017).
- D. J. Wu, *Accelerating Self-Play Learning in Go*, arXiv:1902.10565 (2020).
- D. J. Wu et al., *Multi-labelled value networks for computer Go*, IEEE ToG 10(4) (2018).
- Y. Tian, Y. Zhu, *Better computer Go player with neural network and long-term prediction*, ICLR (2016).
- J. Hu et al., *Squeeze-and-Excitation Networks*, CVPR (2018).
- D. B. Benson, *Life in the game of Go*, Information Sciences 10 (1976).

## 许可

MIT。仅用于学习与复现研究，不附带任何棋力保证。
