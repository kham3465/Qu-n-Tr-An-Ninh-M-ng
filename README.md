# SlaTriage

Hội đồng LoRA **R / A / C + Judge** phân xử cảnh báo Slither: mỗi alert quyết **keep** (giữ) hoặc **drop** (bỏ).

Không viết detector mới. Không săn lỗ từ trang trắng kiểu LLM-SmartAudit. Đầu vào là JSON Slither; đầu ra là quyết định trên từng cảnh báo đã có.

**Trạng thái (2026-10-06):** pipeline Curated đã chạy xong, nhãn + SFT đã xuất. **Bước tiếp theo là train QLoRA trên GPU** (máy local CUDA hoặc Kaggle T4). Máy Windows hiện tại chỉ có `torch` CPU — chưa train được.

---

## 1. Bài toán

Slither bắn nhiều cảnh báo. Auditor thật gặp hai lỗi ngược nhau:

- Tool **ồn**: nhiều alert không phải lỗ (false positive).
- LLM một mình **ảo**: dễ bịa loại lỗ hoặc số dòng không có trong file (*invent*).

SlaTriage làm **adjudication / triage**: với từng alert, quyết giữ hay bỏ. Detector (`reentrancy-eth`, `tx-origin`, …) là tên quy tắc của Slither, dùng để **xếp họ**, không phải thứ bị keep/drop.

| Khái niệm | Nghĩa |
| --- | --- |
| **Detector** | Tên quy tắc Slither (trường `check`) |
| **Alert** | Một lần quy tắc bắn: detector + file + dòng + mô tả |
| **keep** | Khớp nhãn người cùng họ và gần dòng (≤ 5) — lỗ thật, giữ |
| **drop** | Không khớp nhãn người — ồn, bỏ |
| **unknown** | Thiếu dòng hoặc khớp nhiều nhãn — không train, không tính F1 |

Không claim “không sót mọi lỗ trên đời”. Durieux et al. (ICSE 2020): 9 tool gộp ~42% lỗ đã gắn nhãn trên Curated. Nhóm đảm bảo **không sót quyết định** trên mọi alert trong thang và mọi hạng DASP đã tuyên bố. Lỗ không có alert máy → ô **ABSENT** (sót có sổ).

---

## 2. Hội đồng 4 tác nhân

Cùng nền `Qwen/Qwen2.5-Coder-7B-Instruct`, mỗi vai một adapter LoRA. Baseline miễn phí: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (cấu hình G, một prompt, không LoRA). Inference **temperature 0**.

| Tác nhân | Việc | Cấm |
| --- | --- | --- |
| **R** | Reentrancy / luồng ether | Bắt `tx.origin` |
| **A** | Access control / `tx.origin` | Bắt reentrancy |
| **C** | Unchecked call / `delegatecall` | Bắt loại ngoài họ |
| **J (Judge)** | Gộp, bác, chốt danh sách cuối | **Tự thêm finding** |

LLM **không** tự chọn họ. `src/slatriage/families.py` tra `configs/detectors_map.yaml`:

| Họ | Detector |
| --- | --- |
| R | `reentrancy-eth`, `reentrancy-no-eth`, `reentrancy-benign`, `reentrancy-events`, `reentrancy-unlimited-gas`, `eth-send` |
| A | `tx-origin`, `controlled-array-length`, `suicidal`, `arbitrary-send-eth`, `arbitrary-send` |
| C | `unchecked-lowlevel`, `unchecked-send`, `unchecked-transfer`, `controlled-delegatecall`, `delegatecall-loop` |
| O (phụ) | `timestamp`, `tautology`, `unused-return` — chưa đủ mẫu lập họ riêng |
| ngoài thang | informational (`solc-version`, `naming-convention`, …) — không train, không F1 chính |

Một họ chỉ được lập khi đủ: (1) Slither sinh alert, (2) có nhãn người keep **và** drop, (3) câu hỏi FP khác họ đã có, (4) đủ mẫu (`min_keep` / `min_drop` = 25 trong `closed_list.yaml`). Trên Slither + Curated hiện sống sót **R, A, C**. Đếm xong cụm thứ tư đủ ngưỡng thì thêm họ; ablation không giảm F1 thì gộp/bỏ.

Khác các bài đã công bố: LLM-SmartAudit = vai chức danh + chat + tìm lỗ từ mã; GPTLens = nhiều auditor cùng prompt; GPTScan = GPT xác nhận lỗ logic; iAudit = fine-tune trên *hàm*. Đơn vị của nhóm là **alert**, vai = **họ lỗ**, học **keep/drop**.

---

## 3. Luồng hệ thống

```text
file .sol
  → copy sang %TEMP% (tránh đường dẫn Unicode làm solc/Slither fail trên Windows)
  → solc khớp versions.csv / pragma; fallback 0.4.25
  → slither --json   (exit ≠ 0 vẫn OK nếu có results.detectors)
  → xếp họ theo detectors_map
  → khớp vulnerabilities.json: cùng họ + |line| ≤ 5 → keep / drop / unknown
  → xuất SFT R/A/C/J
  → (GPU) QLoRA từng adapter
  → hội đồng A–E–G + Judge cấm invent
  → F1 + ma trận đóng KEEP|DROP|ABSENT|UNKNOWN
```

Sáu cấu hình so sánh:

| ID | Hệ |
| --- | --- |
| **A** | Chỉ Slither (mọi alert in-scope = keep) |
| **B** | Qwen-7B, một prompt, chưa train |
| **C** | Bốn vai chỉ prompt, chưa LoRA |
| **D** | 3 LoRA + Judge prompt |
| **E** | 3 LoRA + Judge đã train (**đề xuất**) |
| **G** | DeepSeek-14B, một prompt, chưa train |

E vs B: hội đồng học có hơn unprompted 7B không.  
E vs C: LoRA có hơn chỉ chia vai prompt không.  
E vs D: train Judge có giảm invent không.  
E vs G: 7B đã học vs 14B thô.

**Invent:** finding không nằm trong danh sách keep của chuyên gia, hoặc `line` không có trong file nguồn. Judge phải để `invented_count`.

Ma trận đóng L = 10 hạng DASP (`configs/closed_list.yaml`). Mỗi (contract × hạng) một ô; không có alert máy thì **ABSENT**, Coverage Officer **cấm KEEP**. Arithmetic / front-running thường ABSENT nếu chỉ Slither — đúng literature.

---

## 4. Kết quả pipeline Curated (đã chạy)

SmartBugs Curated, 143 file `.sol`:

- Slither OK: **141/143**
- Fail (bug Slither 0.11, assert tuple-assign): `reentrancy_bonus.sol`, `reentrancy_cross_function.sol`
- Alert in-scope R/A/C: **343**
- SFT (đã loại unknown): R=164, A=77, C=83, J=112

| Họ | keep | drop | unknown |
| --- | ---: | ---: | ---: |
| R | 28 | 136 | 3 |
| A | 9 | 68 | 1 |
| C | 56 | 27 | 15 |

Keep theo detector: `unchecked-lowlevel` 46, `reentrancy-eth` 27, `unchecked-send` 10, `controlled-array-length` 3, `arbitrary-send-eth` 2, `tx-origin` 2, `suicidal` 2, `reentrancy-no-eth` 1.

Họ A keep mỏng vì nhiều bug access-control Curated (sai tên constructor, ghi mapping) **không** trùng detector Slither A — đúng ô ABSENT, không phải sót keep. Train A vẫn được (77 dòng keep+drop) nhưng lệch lớp; ghi rõ trên báo.

File kết quả local (không commit lên git, xem `.gitignore`):

- `data/labels/labels_v1.jsonl`
- `data/labels/stats.md`
- `adapters/{R,A,C,J}/sft_*.jsonl`
- `data/raw/slither_json/`
- `reports/coverage_matrix.jsonl`

---

## 5. Cài môi trường (CPU / data)

```bash
cd slatriage
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install slither-analyzer solc-select
python scripts/smoke_test.py
python -m pytest tests -q
```

Clone dữ liệu (nếu máy chưa có):

```bash
cd data/raw
git clone https://github.com/smartbugs/smartbugs-curated.git
cd ../..
```

Chạy lại toàn bộ Curated → nhãn → SFT (**không** train):

```bash
python scripts/prepare_curated.py
```

Một file lẻ:

```bash
python scripts/ingest_sol.py --sol path/Contract.sol --out-dir data/raw/slither_json
```

Ma trận coverage từ nhãn đã có:

```bash
python scripts/build_coverage.py --alerts data/labels/labels_v1.jsonl --out reports/coverage_matrix.jsonl
```

Không dùng HuggingFace `slither-audited-*` làm nhãn (học vẹt chính tool đang triage).

---

## 6. Train QLoRA — đến lúc cần GPU

Base: `Qwen/Qwen2.5-Coder-7B-Instruct`  
QLoRA 4-bit, `r=16`, `lora_alpha=32`, `q_proj`/`v_proj` (`configs/models.yaml`).  
Cần khoảng **≥ 10 GB VRAM** cho 7B. Thứ tự: **R → A → C → J**. Bỏ `unknown`.

### 6.1 Máy GPU local

```bash
python scripts/check_env.py
# cần cuda_available=True, transformers, peft, trl, bitsandbytes
# torch CUDA: https://pytorch.org

python scripts/train_lora.py --role R --data data/labels/labels_v1.jsonl --do-train
python scripts/train_lora.py --role A --data data/labels/labels_v1.jsonl --do-train
python scripts/train_lora.py --role C --data data/labels/labels_v1.jsonl --do-train
python scripts/train_lora.py --role J --data data/labels/labels_v1.jsonl --do-train
```

Adapter ghi `adapters/{R,A,C,J}/`. Không commit `.safetensors`.

### 6.2 Kaggle — dùng được

**Kết luận: train 4 LoRA 7B trên Kaggle được.** Infer DeepSeek-14B 4-bit cũng vừa T4 16 GB nếu batch=1, nhưng chật hơn.

| Hạng mục | Thực tế Kaggle (2026) |
| --- | --- |
| GPU miễn phí | Tesla **T4 16 GB** (đôi khi P100 16 GB) |
| Quota | khoảng **30 giờ GPU / tuần** / account; session ~9–12 giờ |
| 7B QLoRA 4-bit | ~8–12 GB — **vừa T4** |
| 14B infer 4-bit (cấu hình G) | ~10–13 GB — vừa, tắt cache dài |
| Internet | bật trong notebook để tải model HuggingFace |
| Dataset | upload `sft_R.jsonl` … `sft_J.jsonl` + `labels_v1.jsonl` (nhẹ) |

Cách làm ngắn:

1. Đẩy repo `slatriage` lên GitHub (không gồm `data/raw`, `adapters/*.safetensors` — đã gitignore).
2. Tạo Kaggle Dataset riêng, upload 4 file SFT + `labels_v1.jsonl` (zip từ máy local).
3. Notebook mới → **Accelerator = GPU T4** → Add dataset + (tuỳ chọn) clone git.
4. Cài gói train rồi gọi đúng `train_lora.py` từng role; mỗi role ~vài phút đến ~1 giờ trên T4 vì SFT nhỏ (77–164 dòng).
5. Sau mỗi role: zip `adapters/X` trong `/kaggle/working` và **Download** — session mất thì mất file.

```python
# cell Kaggle (minh họa)
!pip install -q peft trl bitsandbytes
!git clone https://github.com/<user>/slatriage.git
# gắn dataset SFT vào /kaggle/input/...

!python slatriage/scripts/train_lora.py --role R \
  --data /kaggle/input/<dataset>/labels_v1.jsonl \
  --out /kaggle/working/adapters/R \
  --do-train
```

Lưu ý Kaggle:

- Phải tương tác / bật session; idle lâu notebook có thể chết.
- `bitsandbytes` trên T4 ổn; nếu lỗi CUDA, pin `bitsandbytes` bản tương thích torch sẵn có của Kaggle.
- Qwen2.5-Coder-7B-Instruct thường **public** — không bắt buộc HF token. Nếu bị rate-limit, thêm secret `HF_TOKEN`.
- Đủ quota 30h/tuần cho 4 adapter + vài lần infer A–E. Đừng infer G lặp lại cả Curated nếu không cần.
- Kaggle **không thay** máy GPU nhóm nếu sau này train lại nhiều lần / Web3Bugs; đủ cho P1 train v1.

`check_env` trên máy hiện tại: `torch=2.11.0+cpu`, `cuda_available=False` → **phải dùng Kaggle hoặc máy CUDA khác**.

---

## 7. Sau khi có adapter — hội đồng và đo

```bash
# mock (không GPU, kiểm pipeline)
python scripts/run_council.py --alerts data/labels/demo_labels.jsonl --config E --backend mock --out reports/pred_E.jsonl

# thật (GPU + adapter)
python scripts/run_council.py --alerts data/labels/labels_v1.jsonl --config E --backend hf ^
  --adapter-r adapters/R --adapter-a adapters/A --adapter-c adapters/C --adapter-j adapters/J ^
  --out reports/pred_E.jsonl

python scripts/eval_configs.py --gold data/labels/labels_v1.jsonl --pred reports/pred_E.jsonl --out reports/summary.json
```

Lặp `--config` A, B, C, D, G. Metric: P/R/F1 trên nhãn `keep` (bỏ unknown). RQ: gain `(P_E - P_B) / P_B` (mốc kiểm 15%, đạt/không đều ghi). Đo % invent ở C, D, E. Ablation: tắt LoRA-R thì F1 họ R phải giảm.

Schema chuyên gia:

```json
{"alert_id": "reentrancy-eth_0", "family": "R", "decision": "keep", "line": 42, "reason": "..."}
```

Schema Judge:

```json
{"findings": [{"alert_id": "reentrancy-eth_0", "detector": "reentrancy-eth", "line": 42}], "invented_count": 0}
```

---

## 8. Cấu trúc repo

```text
configs/
  detectors_map.yaml    detector → họ R/A/C/O
  closed_list.yaml      10 hạng DASP + ngưỡng lập họ
  models.yaml           Qwen-7B, DeepSeek-14B, LoRA
src/slatriage/
  families.py           assign_family
  label_match.py        khớp alert–nhãn người
  curated.py            vulnerabilities.json, solc snap
  slither_runner.py     chạy Slither (ASCII temp, JSON dù exit ≠ 0)
  coverage.py           KEEP|DROP|ABSENT|UNKNOWN
  council.py            cấu hình A–E–G
  dataset.py            xuất SFT
  prompts.py            prompt keep/drop, Judge cấm invent
scripts/
  prepare_curated.py    Curated → labels_v1 → SFT (không train)
  train_lora.py         --prepare-only | --do-train
  run_council.py        infer hội đồng
  eval_configs.py       P/R/F1
  build_coverage.py     ma trận đóng
  check_env.py          CUDA / gói train
data/labels/            labels_v1.jsonl (local), demo_labels.jsonl (trong git)
adapters/               sft_*.jsonl + adapter sau train (local)
docs/                   outline báo, metric
```

Git: đẩy **thư mục `slatriage/`** (đã `git init`, branch `master`). Không đẩy cả folder môn học. `.gitignore` loại `data/raw/**`, `data/labels/*.jsonl` (trừ demo), `adapters/**`, `reports/**`.

---

## 9. Việc còn lại

1. **Train 4 LoRA trên GPU / Kaggle** (đang chặn).
2. Infer A–E–G trên `labels_v1`, tính F1 + invent + ablation.
3. (Tuỳ chọn) 110 ca LLM-SmartAudit, Web3Bugs nếu biên dịch được; &lt;20 file thì ghi “chưa đủ mẫu”.
4. Viết báo hội nghị: triage cảnh báo, không “audit multi-agent như SmartAudit”. Dàn ý: `docs/outline-bai-bao-hoi-nghi-vn.md`.
