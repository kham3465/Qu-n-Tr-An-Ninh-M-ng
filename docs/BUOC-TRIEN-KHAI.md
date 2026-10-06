# Các bước triển khai SlaTriage

**Lịch nhóm: 3 tuần — đủ phạm vi, không cắt hạng mục.** Map bước → tuần:

| Bước dưới đây | Tuần | Ghi chú |
| --- | --- | --- |
| 0–3 (env, clone, Slither 20 → Curated, map) | **Tuần 1** | Song song tải Qwen + DeepSeek |
| 4 (labels Curated + 110 ca + chấm ≥100) | **Tuần 1 cuối → Tuần 2 đầu** | Freeze v1 trước train full |
| 5 (train R/A/C/J) | **Tuần 2** | Train chính |
| 6 (A–E–G + invent + ablation) | **Tuần 2 cuối → Tuần 3** | Đủ 6 cấu hình |
| 7 (Web3Bugs RQ2) | **Tuần 2–3** | Song song; thiếu mẫu thì ghi rõ |
| 8 (viết báo) | **Tuần 3** | Song song với eval |

Làm **theo thứ tự phụ thuộc** (nhãn trước train full). Chi tiết lịch ngày: [`../../KE-HOACH-TRIEN-KHAI.md`](../../KE-HOACH-TRIEN-KHAI.md).

---

## Bước 0 — Môi trường (cả nhóm, ngày 1)

```bash
cd slatriage
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
pip install slither-analyzer solc-select
```

P2 kiểm tra GPU:

```bash
nvidia-smi
```

Ghi VRAM vào kế hoạch nhóm. Cài đúng bản `torch` CUDA nếu cần: https://pytorch.org

**Done:** `import torch; torch.cuda.is_available()` = True (máy train).

---

## Bước 1 — Clone dữ liệu (P1 + P3)

```bash
cd data/raw
git clone https://github.com/smartbugs/smartbugs-curated.git
git clone https://github.com/LLMAudit/LLMSmartAuditTool.git
# Tuần sau / RQ2:
# git clone https://github.com/ZhangZhuoSJTU/Web3Bugs.git
cd ../..
```

**Done:** có thư mục Curated trong `data/raw/`.

---

## Bước 2 — Chạy Slither 20 file thử (P1)

Chọn solc khớp pragma (ví dụ 0.4.25 / 0.8.x tùy file):

```bash
solc-select install 0.8.20
solc-select use 0.8.20
```

```bash
python scripts/run_slither_batch.py ^
  --input data/raw/smartbugs-curated/dataset ^
  --out data/raw/slither_json ^
  --limit 20
```

Đọc `data/raw/slither_json/summary.json`: tỷ lệ `ok`, số alert.

**Cổng:** nếu >½ fail compile → sửa toolchain, **chưa** làm bước 4–5.

**Done:** ≥10/20 file ra JSON.

---

## Bước 3 — Chốt map detector → R/A/C (P1, cả nhóm review)

Sửa `configs/detectors_map.yaml` cho khớp detector thật trong JSON Slither.

- R: reentrancy / gửi ether  
- A: tx.origin / access control tool bắt được  
- C: unchecked call / delegatecall  
- Ngoài map → **không** vào thang chính

**Done:** cả nhóm OK map; commit file yaml.

---

## Bước 4 — Dựng nhãn keep/drop (P1) — *bắt buộc trước train*

1. Chạy Slither **toàn** Curated (file compile được), bỏ `--limit` hoặc tăng limit.  
2. Khớp với `vulnerabilities.json` (Curated):

```bash
python scripts/build_labels.py ^
  --slither-dir data/raw/slither_json ^
  --vuln-json data/raw/smartbugs-curated/.../vulnerabilities.json ^
  --map configs/detectors_map.yaml ^
  --out data/labels/labels.jsonl
```

3. Chấm tay ≥100 alert → `data/labels/manual_sample.csv` (P1 + P3).  
4. **Freeze labels v1** (không sửa lung tung lúc train).

**Done:** có `labels.jsonl` + đếm keep/drop/unknown theo họ R/A/C.

---

## Bước 5 — Train LoRA hội đồng (P2)

Base: `Qwen/Qwen2.5-Coder-7B-Instruct` (xem `configs/models.yaml`).

```bash
python scripts/train_lora.py --role R --data data/labels/labels.jsonl
python scripts/train_lora.py --role A --data data/labels/labels.jsonl
python scripts/train_lora.py --role C --data data/labels/labels.jsonl
python scripts/train_lora.py --role J --data data/labels/labels.jsonl
```

*(Script hiện là skeleton — P2 nối QLoRA/peft trước khi chạy full.)*

Thứ tự: R → A → C → Judge (SFT hoặc DPO cấm invent).

**Done:** adapter trong `adapters/R|A|C|J/`.

---

## Bước 6 — Chạy cấu hình A–E–G (P2 sinh pred, P3 chấm)

| ID | Việc |
| --- | --- |
| A | Mọi alert Slither = keep |
| B | Qwen-7B, 1 prompt, chưa LoRA |
| C | 4 vai prompt, chưa LoRA |
| D | 3 LoRA + Judge prompt |
| E | 3 LoRA + Judge đã train |
| G | DeepSeek-R1-Distill-Qwen-14B, 1 prompt, chưa train |

Nhiệt độ **0**. Cùng schema JSON (`docs/metrics.md`).

```bash
python scripts/eval_configs.py ^
  --gold data/labels/labels.jsonl ^
  --pred reports/pred_A.jsonl reports/pred_B.jsonl reports/pred_E.jsonl reports/pred_G.jsonl ^
  --out reports/summary.json
```

Đo thêm: % invent Judge ở C/D/E; ablation tắt từng LoRA.

**Done:** `reports/summary.json` + bảng invent.

---

## Bước 7 — RQ2 / mở rộng (bắt buộc có câu trả lời trên báo)

Chạy **song song tuần 2–3**, không chờ xong hết eval mới làm:

- Thêm nhãn 110 ca + nhóm sạch (LLM-SmartAuditTool) — **tuần 2 đầu**, trước/alongside train  
- Lát Web3Bugs biên dịch được (20–40 file) → RQ2  
- Nếu <20 file compile: ghi “chưa đủ mẫu” + số đã thử — **không bỏ RQ2**

---

## Bước 8 — Viết báo + đóng gói (tuần 3, song song E/G)

| Ai | Mục |
| --- | --- |
| P1 | Data, thống kê compile / unknown |
| P2 | Hyperparam, giờ GPU, lệnh reproduce |
| P3 | Method, Eval, Related Work, bảng A–E–G |

Release: labels (hoặc script sinh lại) + configs + scripts — **không** nhét checkpoint nặng lên git.

---

## Luồng một hợp đồng (nhắc nhanh)

```text
.sol → solc đúng pragma
    → Slither --json
    → chia alert theo R / A / C
    → LoRA chuyên gia (họ trống thì bỏ qua)
    → Judge (cấm invent)
    → báo cáo JSON
```

## Cổng không được phá

1. Không train trên nhãn = output Slither (HF slither-audited).  
2. Không train full trước freeze labels v1.  
3. Không claim đã tái lập MOS / GPT-4 TA trên tập họ.
