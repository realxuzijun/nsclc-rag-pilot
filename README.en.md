# Lung radiotherapy protocol RAG handoff

[简体中文](README.md)

This research prototype answers questions about public lung radiotherapy protocols. It queries structured dose, image verification, and contouring tables in SQLite. It retrieves, reranks, and verifies individual clauses for narrative questions. Both paths show the original evidence and its location. For a new collection, start with text retrieval and add structured tables once the document layout and terminology are clear.

## 1. Set up the runtime and models

Run these Linux Bash commands from the project root. The original experiment used Python 3.8.10, and the fresh isolated setup used Python 3.11.18. Both used Ollama 0.20.5 and one RTX A6000. Query scripts use the Python standard library and SQLite. Install the two packages below if you need to extract the original HTML and PDFs. Building the reranker also requires CMake, a C/C++ compiler, and build tools.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install beautifulsoup4 pdfplumber
~~~

If the system Python lacks `ensurepip`, install its matching `python3-venv` package or use [uv](https://docs.astral.sh/uv/getting-started/installation/) to install Python inside the project directory without administrator access. These are the Python and parser versions used for the clean setup.

~~~bash
curl -fsSLo uv-install.sh https://astral.sh/uv/install.sh
UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh uv-install.sh
export UV_CACHE_DIR="$PWD/.uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/.python"
.tools/uv venv --python 3.11 .venv-uv
source .venv-uv/bin/activate
.tools/uv pip install beautifulsoup4==4.15.0 pdfplumber==0.11.11
~~~

Install Ollama using the [official Linux instructions](https://docs.ollama.com/linux). You can reuse a running instance. To install the reference version in isolation on Linux x86-64, unpack the [official v0.20.5 release](https://github.com/ollama/ollama/releases/tag/v0.20.5) into the project directory. This requires `curl`, `tar`, and `zstd`. The runtime archive is about 2 GB and the initial model downloads add about 6.4 GB.

On Ubuntu without `zstd` or administrator access, run `apt download zstd` and `dpkg-deb -x ./zstd_*.deb .tools/zstd` from the project directory, then run `export PATH="$PWD/.tools/zstd/usr/bin:$PATH"`.

~~~bash
mkdir -p ollama-runtime
curl -fL --retry 2 https://github.com/ollama/ollama/releases/download/v0.20.5/ollama-linux-amd64.tar.zst -o ollama-runtime/ollama-linux-amd64.tar.zst
printf '%s  %s\n' 297fd50c2349b8bc83f5c87eae757381c1532c10fe0a813064e6814cdb9c90bb ollama-runtime/ollama-linux-amd64.tar.zst | sha256sum -c
tar --zstd -xf ollama-runtime/ollama-linux-amd64.tar.zst -C ollama-runtime
./ollama-runtime/bin/ollama --version
~~~

Start the service and pull the models in separate terminals. If you used the local installation, first run `export PATH="$PWD/ollama-runtime/bin:$PATH"` in both terminals. If you are reusing an existing model service, point the query scripts at it and skip launching a new instance.

~~~bash
# Terminal A
OLLAMA_HOST=127.0.0.1:11543 OLLAMA_MODELS="$PWD/model-store" ollama serve

# Terminal B
export OLLAMA_HOST=127.0.0.1:11543
ollama pull qwen3:8b
ollama pull bge-m3
curl -fsS http://127.0.0.1:11543/api/tags
~~~

| Role | Model and reference version |
|---|---|
| Chinese explanation, relevance checking, and table layout | [Qwen3-8B](https://ollama.com/library/qwen3:8b), `qwen3:8b`, Q4_K_M, digest prefix `500a1f067a9f` |
| Embeddings for questions and clauses | [BGE-M3](https://ollama.com/library/bge-m3), `bge-m3`, F16, digest prefix `790764642607` |
| Independent reranking | BGE reranker v2 m3 Q8_0 served by llama.cpp on CPU in the reference setup |

If the handoff includes a working `reranker_runtime/`, check that `runtime.path` points to the local `llama-server` binary and start the service. For a fresh setup, build [llama.cpp b12608](https://github.com/ggml-org/llama.cpp/releases/tag/b12608) and download the weights from the [pinned revision](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/tree/3093af03b1a635e67b084b1d8c03c5f5e020fd05). If system CMake is unavailable, unpack the [Kitware CMake 3.31.8](https://github.com/Kitware/CMake/releases/tag/v3.31.8) archive for your architecture and add its `bin` directory to the terminal's `PATH`.

~~~bash
mkdir -p reranker_runtime
curl -fL --retry 2 https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b12608 -o reranker_runtime/llama.cpp-b12608.tar.gz
tar -xzf reranker_runtime/llama.cpp-b12608.tar.gz -C reranker_runtime
cmake -S reranker_runtime/llama.cpp-b12608 -B reranker_runtime/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_NATIVE=ON
cmake --build reranker_runtime/build --config Release --target llama-server -j 4
printf '%s\n' "$(realpath reranker_runtime/build/bin/llama-server)" > reranker_runtime/runtime.path
curl -fL https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/3093af03b1a635e67b084b1d8c03c5f5e020fd05/bge-reranker-v2-m3-Q8_0.gguf -o reranker_runtime/bge-reranker-v2-m3-Q8_0.gguf

# Terminal C. The launcher verifies the weights' SHA-256 before starting.
bash start_reranker.sh
~~~

From another terminal, check the reranker with `curl -fsS http://127.0.0.1:11544/v1/models`. Use `--base` and `--rerank-base` with `protocol_answer.py`, `round8.py`, or `round9.py` to point to existing services. Pass the Ollama service root without a trailing slash or `/api`.

If these ports are already in use, give the new Ollama instance another `OLLAMA_HOST`, start the reranker with `RERANK_PORT=12544 bash start_reranker.sh`, and pass the corresponding service URLs to the query and evaluation scripts.

To try a different generation or embedding model, update `MODEL` or `EMBED` in `benchmark.py`. The generation model must support Ollama's JSON structured output. Rerun the smoke tests and evaluation, and record the digest reported by `/api/tags`.

## 2. Download the public test sources and build the data

[public_sources.json](public_sources.json) lists the official links, reference versions, and original snapshot hashes for the eviQ, NCI PDQ, West Midlands, and Wake Forest sources. The download helper puts those sources under `data/raw/`. You can also save the documents yourself using the paths in the manifest or copy in the original handoff snapshots. Follow the [eviQ reuse terms](https://www.eviq.org.au/copyright) and the [NCI reuse policy](https://www.cancer.gov/policies/copyright-reuse).

~~~bash
python fetch_public_data.py
python build_corpus.py
python build_structured.py
python extract_transfer_protocols.py
python extract_publisher_tables.py
python build_protocol_tables.py
~~~

The download helper reports differences from the reference hashes and keeps existing local files for review. A publisher may have changed the live document or layout. For an exact reproduction, use the reference snapshots and compare their hashes to the manifest. The PDF import covers selected pages and tables, including the West Midlands dose and contour tables and the Wake Forest conventional fractionation table. The two generated SQLite databases and all review outputs stay under ignored local directories.

## 3. Smoke-test both paths and reproduce the evaluation

~~~bash
# SQL missing-value case. Expect an original dash, PDF page 14 and a separate header link.
python protocol_answer.py --question 'West Midlands 胸壁 Dmax (0.1 cc) 在 8 次分割的强制栏写了限值吗？'

# Narrative search. Expect the eviQ 1871 simulation clause and a Chinese explanation.
python protocol_answer.py --question 'eviQ 1871 模拟定位的计划 CT 扫描范围，从哪里开始到哪里结束？' --terms 'simulation planning CT scan cranial caudal limits'

mkdir -p results
python round9.py
~~~

Round 9 runs both routes on 15 questions, including five missing-item cases. Review the output in `results/round9.json` against `round9_gold.json`. Each run updates that local result file. Save a separate copy to retain a run.

## 4. Add your own data

### A. Start with text retrieval

1. Retain each original document and record its publisher, version, URL or internal document reference, and SHA-256. Keep PDF page numbers or HTML section anchors. Split by clause and keep conditions, exceptions, and applicability together.
2. Convert the clauses to UTF-8 JSONL and append them to `data/external_clauses.jsonl`. Each `id` must be unique across both clause files. The `source` must match the ID selected by the question router. For a quick test, use an unused four-digit string and ask with an explicit label such as `协议编号 9901`. To support a custom name or alphanumeric identifier, update `protocol_scope()` in `citation_guard.py`.
3. Run `protocol_answer.py`. The current CLI embeds clauses in the selected scope on each invocation, so a text-only import needs no SQLite rebuild. Put your custom import or merge after `extract_transfer_protocols.py` because that script rewrites the external clause file. For a larger collection, consider building and reusing an offline index.

This synthetic example exercises the existing text route. Run it once in a project copy and ensure the JSONL file ends with a newline before appending.

~~~bash
cat >> data/external_clauses.jsonl <<'JSONL'
{"id":"9901-v1-p2-L001","source":"9901","title":"Synthetic handoff example","version":"v1","section":"Quality assurance","url":"https://example.org/protocol-9901.pdf#page=2","text":"The designated medical physicist must review the quality assurance record before the first session."}
JSONL
python protocol_answer.py --question '协议编号 9901 的质控记录由谁复核？' --terms 'quality assurance record review'
~~~

Expect a citation to `9901-v1-p2-L001` and a Chinese answer identifying the designated medical physicist. Replace the placeholder URL with a working source link for real data.

### B. Add structured tables

1. Write an extractor and field mapping for the PDF, HTML, CSV, or workbook. Use `extract_publisher_tables.py` and `data/publisher_tables.json` as examples. For scanned PDFs, run OCR and manually verify the pages, continued headers, symbols, units, and footnotes. Map dose entries to organ, metric, fractionation, tier, volume definition, comparison operator, value, and unit. Store image verification modality, frequency, timing, and registration target separately.
2. Link each fact to a source version and evidence ID. Distinguish a printed dash, an unprinted field, and a table outside the imported scope. Keep missing units and comparison operators null and preserve the literal cell.
3. To reuse the PDF dose or contour schema, add the source, source rows, and normalized records to `publisher_tables.json`. Update `publisher_tables.py` with the alias, display name, and lookup logic, and point missing-value citations to **the new protocol's own table header or inventory**. Its PDF routing is specific to the two existing protocols, so appending data alone will not activate a new structured route. To add image verification, follow `imaging_rows()` in `build_protocol_tables.py` and the verification branch of `table_answers.py`. If extending the eviQ-style dose database, update `build_structured.py` and the dose lookup in `round3.py`.
4. Run `python build_protocol_tables.py` to regenerate the shared database. Check exact values, applicability, multi-metric questions, missing items, and similar clauses from other protocols. Rerun the original questions as a regression check. Keep scoring references separate and compare answers against the source. `round9.py` is specific to the original test protocols and can serve as a template for a custom evaluation.

## 5. Troubleshooting

| Symptom | What to check first |
|---|---|
| Connection refused or model missing | Check `/api/tags` and confirm the model was pulled into the instance the scripts use. For text queries, check the reranker's `/v1/models` as well |
| Large downloads time out on the server | Download the same pinned archive or weights on a workstation with network access and copy them to the server. Verify the SHA-256 with the installation command or reranker launcher. If an Ollama model pull is interrupted, repeat `ollama pull` against the same instance |
| An unrelated protocol appears among the candidates | Ask with an explicit protocol ID, verify the clause's `source` and the routing aliases, and inspect the candidate and citation audit records |
| A table, row, or field is missing | Check the database timestamp, the original cells and footnotes, and the lookup's organ and metric aliases |
| Custom clauses disappear after a rebuild | Run the custom merge after the extraction step. Persist custom PDF records in the extractor used to regenerate them |
| Latency or GPU memory is high | Check `/api/ps` and `nvidia-smi`. Separate cold loading, on-demand embeddings, CPU reranking, and generation |

For handoff, first reproduce the two smoke tests, then load one document into a copy of the project. This prototype was evaluated for literature question answering and evidence citation. Clinical use requires separate validation.
