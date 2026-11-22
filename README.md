# 肺癌放疗协议 RAG

[English](README.en.md)

这是公开肺癌放疗协议问答的研究原型。已结构化的剂量、影像验证和勾画表走 SQLite，正文走条款检索、重排序及引用校验。两条路径都附原文和出处。新项目可以先接入正文，等数据和术语稳定后再结构化表格。

## 硬件配置与显存

### 实测显存

在单张 NVIDIA RTX A6000（48 GB）上，使用 Ollama 0.20.5 和 Qwen3-8B Q4_K_M，生成请求的上下文设为 8,192 tokens。独立重排序服务在 CPU 上运行，配置为 8 线程。两条路径各运行一道题，用 `nvidia-smi` 每 100 毫秒采样整卡占用。

| 路径 | 整卡显存峰值 |
| --- | ---: |
| SQLite 查表 + 中文排版，Qwen 冷启动 | 6,316 MiB（6.17 GiB） |
| 正文检索 + 中文解释，Qwen 已加载 | 8,003 MiB（7.82 GiB） |

这组数据适用于上述单请求配置。并发请求、长上下文和其他量化规格需要单独测量。

### 配置建议

下表是按本项目的小型语料库和上述峰值给出的容量建议。

| 场景 | GPU 显存 | CPU | 系统内存 | SSD 可用空间 |
| --- | ---: | ---: | ---: | ---: |
| 单人开发和复现实验 | 12 GiB | 8 个逻辑线程 | 16 GiB | 25 GB |
| 更大的语料库或多服务同机 | 16 GiB 或以上 | 8–16 个逻辑线程 | 32 GiB 或以上 | 40 GB 或以上 |

8 GiB 显卡在实测检索峰值下只剩约 189 MiB 余量，显卡同时承担桌面显示或其他进程时尤其容易吃紧。安装前可核对 GPU 和驱动是否受 Ollama 支持。开启并发或提高上下文前，按实际负载重新采样显存，并用 `ollama ps` 检查模型在 CPU 和 GPU 上的分配。[Ollama 硬件支持](https://docs.ollama.com/gpu) · [上下文长度说明](https://docs.ollama.com/context-length)

## 1. 环境和模型

命令以 Linux Bash 为例，从项目根目录运行。原实验使用 Python 3.8.10，隔离重装验证使用 Python 3.11.18。两次都使用 Ollama 0.20.5 和单张 RTX A6000。查询脚本用 Python 标准库和 SQLite。仅重抽原始 HTML 或 PDF 时需要下面两个包。重新编译重排序服务时另需 CMake、C/C++ 编译器和构建工具。

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install beautifulsoup4 pdfplumber
~~~

如果系统 Python 缺少 `ensurepip`，可安装对应的 `python3-venv` 软件包，也可用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 在项目目录安装独立 Python，无需管理员权限。以下是实测使用的 Python 与解析器版本。

~~~bash
curl -fsSLo uv-install.sh https://astral.sh/uv/install.sh
UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh uv-install.sh
export UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.python"
.tools/uv venv --python 3.11 .venv-uv
source .venv-uv/bin/activate
.tools/uv pip install beautifulsoup4==4.15.0 pdfplumber==0.11.11
~~~

按 [Ollama Linux 安装文档](https://docs.ollama.com/linux) 安装 Ollama。已有服务可直接复用。要复现基准版本并隔离安装，在 Linux x86-64 上可把 [v0.20.5 官方包](https://github.com/ollama/ollama/releases/tag/v0.20.5) 解压到项目目录。该方式需要 `curl`、`tar` 和 `zstd`。压缩包约 2 GB，首次还需从 Ollama 下载约 6.4 GB 模型。

在 Ubuntu 上没有 `zstd` 或管理员权限时，可以在项目目录执行 `apt download zstd` 和 `dpkg-deb -x ./zstd_*.deb .tools/zstd`，再执行 `export PATH="$PWD/.tools/zstd/usr/bin:$PATH"`。

~~~bash
mkdir -p ollama-runtime
curl -fL --retry 2 https://github.com/ollama/ollama/releases/download/v0.20.5/ollama-linux-amd64.tar.zst -o ollama-runtime/ollama-linux-amd64.tar.zst
printf '%s  %s\n' 297fd50c2349b8bc83f5c87eae757381c1532c10fe0a813064e6814cdb9c90bb ollama-runtime/ollama-linux-amd64.tar.zst | sha256sum -c
tar --zstd -xf ollama-runtime/ollama-linux-amd64.tar.zst -C ollama-runtime
./ollama-runtime/bin/ollama --version
~~~

在两个终端分别启动服务和下载模型。若使用上面的本地安装，在两个终端先执行 `export PATH="$PWD/ollama-runtime/bin:$PATH"`。使用已有模型服务时，直接指定它的地址并跳过启动步骤。

~~~bash
# 终端 A
OLLAMA_HOST=127.0.0.1:11543 OLLAMA_MODELS="$PWD/model-store" ollama serve

# 终端 B
export OLLAMA_HOST=127.0.0.1:11543
ollama pull qwen3:8b
ollama pull bge-m3
curl -fsS http://127.0.0.1:11543/api/tags
~~~

| 用途 | 模型与基准版本 |
|---|---|
| 中文解释、相关性复核和表格排版 | [Qwen3-8B](https://ollama.com/library/qwen3:8b)，`qwen3:8b`，Q4_K_M，摘要前缀 `500a1f067a9f` |
| 中文问题与条款的向量检索 | [BGE-M3](https://ollama.com/library/bge-m3)，`bge-m3`，F16，摘要前缀 `790764642607` |
| 独立重排序 | BGE reranker v2 m3 Q8_0，由 llama.cpp 提供，基准服务使用 CPU |

如果已有可用的 `reranker_runtime/`，核对 `runtime.path` 指向本机的 `llama-server`，直接执行下方启动命令。全新环境可按以下命令构建。没有系统级 CMake 时，可解压 [Kitware CMake 3.31.8](https://github.com/Kitware/CMake/releases/tag/v3.31.8) 对应架构的压缩包，将其 `bin` 加入当前终端的 `PATH`。权重为 [固定提交](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/tree/3093af03b1a635e67b084b1d8c03c5f5e020fd05)，二进制基于 [llama.cpp b12608](https://github.com/ggml-org/llama.cpp/releases/tag/b12608)。

~~~bash
mkdir -p reranker_runtime
curl -fL --retry 2 https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b12608 -o reranker_runtime/llama.cpp-b12608.tar.gz
tar -xzf reranker_runtime/llama.cpp-b12608.tar.gz -C reranker_runtime
cmake -S reranker_runtime/llama.cpp-b12608 -B reranker_runtime/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_NATIVE=ON
cmake --build reranker_runtime/build --config Release --target llama-server -j 4
printf '%s\n' "$(realpath reranker_runtime/build/bin/llama-server)" > reranker_runtime/runtime.path
curl -fL https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/3093af03b1a635e67b084b1d8c03c5f5e020fd05/bge-reranker-v2-m3-Q8_0.gguf -o reranker_runtime/bge-reranker-v2-m3-Q8_0.gguf

# 终端 C，启动脚本先校验权重 SHA-256
bash start_reranker.sh
~~~

在另一个终端执行 `curl -fsS http://127.0.0.1:11544/v1/models` 检查重排序服务。`protocol_answer.py`、`round8.py` 和 `round9.py` 支持用 `--base` 和 `--rerank-base` 指向现有服务。`--base` 填服务根地址，省略末尾斜杠和 `/api`。

已有服务占用这两个端口时，为新 Ollama 实例设置另一个 `OLLAMA_HOST`，用 `RERANK_PORT=12544 bash start_reranker.sh` 启动新重排序实例，并在查询与评测命令中传入对应地址。

更换生成或向量模型时，在 `benchmark.py` 更新 `MODEL` 或 `EMBED`。生成模型需支持 Ollama 的 JSON 结构化输出。换模型后重新跑冒烟和对照，并记录 `/api/tags` 返回的摘要。

## 2. 下载公开测试资料并构建数据

[来源清单](public_sources.json)列出 eviQ、NCI PDQ、West Midlands 和 Wake Forest 的官方链接、参考版本及原始快照摘要。下载脚本会把资料保存到 `data/raw/`。也可以按清单的文件名手工保存原件，或使用原始快照。参见 [eviQ 使用条款](https://www.eviq.org.au/copyright)和 [NCI 使用条款](https://www.cancer.gov/policies/copyright-reuse)。

~~~bash
python fetch_public_data.py
python build_corpus.py
python build_structured.py
python extract_transfer_protocols.py
python extract_publisher_tables.py
python build_protocol_tables.py
~~~

下载程序会报告与参考摘要的差异，并保留已有本地文件供核对。若 West Midlands 的官方 PDF 返回 403，请从能访问官网的网络下载，或取用同版原件，保存为 `data/raw/wmrn.pdf` 后重新运行下载命令。发布方可能更新了在线版本或表格布局。若需要精确复现实验，请使用参考快照，并对照来源清单核验摘要。PDF 只导入选定的页和表，包括 West Midlands 的剂量和勾画表，以及 Wake Forest 的常规分割约束表。生成的 SQLite 和测试输出均保存在忽略提交的本地目录。

## 3. 冒烟测试与复现实验

~~~bash
# SQL 缺项题，应展示横线、PDF 第 14 页和跨页表头
python protocol_answer.py --question 'West Midlands 胸壁 Dmax (0.1 cc) 在 8 次分割的强制栏写了限值吗？'

# 正文检索题，应展示 eviQ 1871 的模拟定位条款，再解释扫描起止
python protocol_answer.py --question 'eviQ 1871 模拟定位的计划 CT 扫描范围，从哪里开始到哪里结束？' --terms 'simulation planning CT scan cranial caudal limits'

mkdir -p results
python round9.py
~~~

第九轮对 15 道题运行两条路径，其中 5 道是缺项题。按 `round9_gold.json` 核对 `results/round9.json` 中的本地输出。每次运行会更新该文件，需要留存时可另行复制。

## 4. 接入自己的数据

### A. 先接正文检索

1. 保存原件，记录发布方、版本、链接或内部文档编号、文件 SHA-256。PDF 保留页码，HTML 保留章节锚点。按条款切分，每条保留条件、例外及适用对象。
2. 把条款转为 UTF-8 JSONL，追加到 `data/external_clauses.jsonl`。`id` 在整个语料库唯一。`source` 必须与题目路由识别的协议编号一致。可先用闲置的四位数字字符串，通过「协议编号 9901」这样的题面激活现有路由。使用自定义名称或字母编号时，修改 `citation_guard.py` 的 `protocol_scope()`。
3. 直接运行 `protocol_answer.py` 测试。当前命令行每次为限定范围的条款计算嵌入，追加正文无需重建 SQLite。后续大语料可增加离线索引。把新条款的生成或合并步骤放在 `extract_transfer_protocols.py` 之后，该脚本会重写外部条款文件。

下面用虚构条款验证接口。仅在项目副本中执行一次，追加前确认 `data/external_clauses.jsonl` 以换行结束。

~~~bash
cat >> data/external_clauses.jsonl <<'JSONL'
{"id":"9901-v1-p2-L001","source":"9901","title":"Synthetic protocol example","version":"v1","section":"Quality assurance","url":"https://example.org/protocol-9901.pdf#page=2","text":"The designated medical physicist must review the quality assurance record before the first session."}
JSONL
python protocol_answer.py --question '协议编号 9901 的质控记录由谁复核？' --terms 'quality assurance record review'
~~~

预期引用 `9901-v1-p2-L001`，中文说明由指定医学物理师复核。示例 URL 是占位符，真实资料应换成可定位的原件链接。

### B. 再接结构化表格

1. 为 PDF、HTML、CSV 或 Excel 编写对应的提取与字段映射，可参照 `extract_publisher_tables.py` 及其生成的 `data/publisher_tables.json`。对于扫描 PDF，先做 OCR，再核对原页、跨页表头、上下标、单位及脚注。已有栏目可沿用，例如剂量按器官、指标、分割方案、档位、计算体积、比较符、数值和单位表达。影像验证另存方式、频率、时点和配准对象。
2. 为每个字段绑定对应版本和证据 ID。区分原表横线、某个字段没写和该表尚未导入。缺单位或比较符保留为空，另外保留原格。
3. 对剂量或勾画采用 PDF 结构时，将新来源、原文行、剂量格或勾画记录纳入 `publisher_tables.json`。同时在 `publisher_tables.py` 注册协议别名和显示名，更新该协议的查值逻辑，并把缺项时的证据指向**该协议自己的表头或表目录**。当前 PDF 路由为现有两份协议编写，只添加数据文件无法自动启用新协议。新增影像验证表时参照 `build_protocol_tables.py` 的 `imaging_rows()` 和 `table_answers.py` 的验证分支。若重用 eviQ 剂量库，则调整 `build_structured.py` 和 `round3.py` 的剂量查值规则。
4. 运行 `python build_protocol_tables.py` 重建共享表库。用新题覆盖正常取值、限定条件、多指标、缺项和相似外协议干扰，再回归原协议题。参考答案单独保存，按原文评分。`round9.py` 是固定协议对照，可作为自定义测试脚本的模板。

## 5. 常见排查

| 现象 | 先检查 |
|---|---|
| 连不上模型或报告模型不存在 | 查 `/api/tags`，确认 `ollama pull` 与 Python 连的是同一 Ollama 实例。正文题也要确认重排序的 `/v1/models` 可用 |
| 服务器下载大文件超时 | 可在网络可达的工作站下载同一固定版本的安装包或权重，再传到服务器。按安装命令或重排序启动脚本核对 SHA-256 后继续。Ollama 模型下载中断时，对同一实例重新执行 `ollama pull` |
| 把别的协议当成候选 | 用明确协议编号提问，核对条款的 `source` 及两个路由模块的别名。查看结果中的候选和引用校验记录 |
| 缺表、缺行或答案漏字段 | 确认使用的是最新 SQLite，检查原表截图、原格、表头和脚注，并对照查询用的器官和指标别名 |
| 新数据重建后消失 | 检查自有条款是否在提取脚本重写输出后重新合并，自定义 PDF 记录是否已写入可重建的适配器 |
| 推理慢或显存高 | 查 `/api/ps` 与 `nvidia-smi`。区分冷启动、正文在线嵌入、CPU 重排序和生成耗时 |

这个原型的验证集中在文献问答及引用，临床使用还需要独立验证。
