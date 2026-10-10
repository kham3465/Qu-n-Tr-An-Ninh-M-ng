# SlaTriage

Phân xử cảnh báo Slither: mỗi **alert** quyết **keep** (giữ, lỗ thật) hoặc **drop** (bỏ, ồn).

Không viết detector mới. Không săn lỗ từ trang trắng kiểu LLM-SmartAudit. Đầu vào là JSON Slither; đầu ra là quyết định trên từng cảnh báo đã có.

Hội đồng bốn vai: ba chuyên gia theo **họ lỗ** R / A / C, plus **Judge** (gộp và cấm bịa finding).

---

## 1. Bài toán

Slither bắn nhiều cảnh báo. Auditor gặp hai lỗi ngược nhau:

- Tool **ồn**: nhiều alert không phải lỗ (false positive).
- Một LLM **ảo**: dễ bịa loại lỗ hoặc số dòng không có trong file (*invent*).

Việc nghiên cứu là **adjudication / triage**, không phải detection.

| Khái niệm | Nghĩa |
| --- | --- |
| **Detector** | Tên quy tắc Slither (`reentrancy-eth`, `tx-origin`, …). Dùng để *xếp họ*, không phải thứ bị keep/drop. |
| **Alert** | Một lần quy tắc bắn: detector + file + dòng + mô tả. |
| **keep** | Cùng họ với nhãn người và gần dòng (cửa sổ 5) — giữ. |
| **drop** | Không khớp nhãn người — bỏ. |
| **unknown** | Thiếu dòng hoặc khớp nhiều nhãn — không học, không tính F1. |

Không claim “không sót mọi lỗ”. Durieux et al. (ICSE 2020): 9 tool gộp khoảng 42% lỗ đã gắn nhãn trên SmartBugs Curated. Nhóm đảm bảo **không sót quyết định** trên mọi alert trong thang và mọi hạng DASP đã tuyên bố. Lỗ không có alert máy → ô **ABSENT** (sót có sổ), không bịa thêm finding.

Khác các bài đã có: LLM-SmartAudit chia vai *chức danh* và tìm lỗ từ mã; GPTLens nhân bản cùng một prompt; GPTScan xác nhận lỗ logic; iAudit fine-tune trên *hàm*. Đơn vị ở đây là **alert**; vai = **họ lỗ**; việc học = **keep/drop**.

---

## 2. Vì sao 3 họ — không 1, không 4

Số họ **không phải tiên đề**. Một họ chỉ được lập khi đủ bốn điều:

1. Slither (hoặc tool trên cùng bus) **sinh được alert**.
2. Curated có **nhãn người** cùng loại, gần dòng, và có cả keep lẫn drop.
3. **Câu hỏi false-positive khác** các họ đã có — nếu cùng một câu thì gộp, không tách.
4. Đủ mẫu sau khi loại unknown (ngưỡng trong `configs/closed_list.yaml`).

Trên Slither + Curated, chỉ **ba cụm** sống sót:

| Họ | Câu hỏi bảo mật (một LoRA = một câu) | Detector điển hình | Cấm |
| --- | --- | --- | --- |
| **R** | Gọi ngoài / chuyển ether trước khi cập nhật state — lỗ hay benign? | `reentrancy-*`, `eth-send` | Bắt `tx.origin` |
| **A** | Quyền gọi / xác thực / gửi ether có bị kiểm sai? | `tx-origin`, `suicidal`, `arbitrary-send-eth` | Bắt reentrancy |
| **C** | Có bỏ qua giá trị trả về, hoặc `delegatecall` tới đích nguy hiểm? | `unchecked-lowlevel`, `unchecked-send`, `controlled-delegatecall` | Bắt loại ngoài họ |

**Vì sao không một model.** Detector đã có tên; từng lớp có phân phối dương tính giả khác nhau (Slither tự tách `reentrancy-eth` vs `reentrancy-benign`; FPR lệch theo loại lỗ trên benchmark). Một đầu ôm hết học lẫn quy tắc. Họ = một cụm detector *cùng câu hỏi*; tắt đúng adapter thì F1 đúng họ phải giảm (ablation). Đó là lý do không dùng vai PM/Auditor: trùng SmartAudit và không khớp detector.

**Vì sao không 4 (hay hơn).** Họ thứ tư phải có tín hiệu máy + nhãn người + câu hỏi FP *mới* + đủ mẫu. `timestamp` / `tautology` / `unused-return` có alert nhưng chưa đủ lập họ → để **O** (thang phụ), không vào F1 chính. Informational (`solc-version`, `naming-convention`) loại khỏi thang. Arithmetic trên Curated gần như không có detector Slither tương ứng → **ABSENT**, không bịa họ “overflow”.

**Vì sao có Judge, không phải họ thứ 4.** Judge không phán một lớp lỗ. Việc của J: chỉ giữ những alert chuyên gia đã keep, bác finding không có trong danh sách đó (*invent*). Đo bằng `invented_count`.

LLM **không** tự chọn họ. `assign_family()` tra `configs/detectors_map.yaml`. Một alert một họ.

Nếu sau này đếm đủ cụm thứ tư theo đúng protocol → thêm họ. Ablation không giảm F1 → gộp hoặc bỏ họ. \(k=3\) là **kết quả protocol trên bằng chứng hiện có**, không phải định lý.

---

## 3. Hướng triển khai

```text
.sol
  → Slither JSON
  → xếp họ R / A / C theo tên detector
  → khớp nhãn người (cùng họ + gần dòng) → keep | drop | unknown
  → học từng chuyên gia + Judge
  → hội đồng A–E–G
  → F1 trên keep + % invent + ma trận đóng
```

Ma trận đóng: 10 hạng DASP (`closed_list.yaml`). Mỗi (hợp đồng × hạng) một ô **KEEP | DROP | ABSENT | UNKNOWN**. Không có alert máy thì ABSENT — cấm ghi KEEP. Hạng không có tín hiệu Slither (arithmetic, front-running, …) để ABSENT, không gộp vào F1 như “đã phát hiện”.

Sáu cấu hình, cùng nhiệt độ 0:

| ID | Hệ | Để trả lời |
| --- | --- | --- |
| **A** | Chỉ Slither (mọi alert in-scope = keep) | Tool thô mạnh đến đâu |
| **B** | Một model, một prompt, chưa học keep/drop | Mốc LLM đơn |
| **C** | Bốn vai, chỉ prompt, chưa LoRA | Chia vai không học có ích không |
| **D** | 3 LoRA + Judge prompt | Học chuyên gia, Judge chưa học |
| **E** | 3 LoRA + Judge đã học (**đề xuất**) | Hệ đầy đủ |
| **G** | Một model lớn hơn, một prompt, không LoRA | Học 3 họ có hơn model thô lớn hơn không |

E vs B: hội đồng đã học có hơn một LLM không.  
E vs C: LoRA có hơn chỉ chia vai không.  
E vs D: học Judge có giảm invent không.  
E vs G: ba họ đã học có hơn một model lớn, không học không.

**Invent:** finding không nằm trong keep của chuyên gia, hoặc dòng không có trong file nguồn.

Quy trình đã làm xong tới hết nhãn:

1. Lấy SmartBugs Curated (`vulnerabilities.json` = nhãn người).
2. Chạy Slither từng file, chuẩn hóa alert.
3. Gán họ bằng bảng detector.
4. Khớp dòng / họ → `data/labels/labels_v1.jsonl`.
5. Xuất mẫu học R, A, C, J (loại unknown).
6. Dựng ma trận coverage từ nhãn.
7. Thêm SolidiFI (bug bơm, chỉ họ R/A/C) làm **tập học phụ** → `labels_solidifi.jsonl` + `data/sft/combined`. F1 chính vẫn Curated.

Chưa làm: học bốn adapter trên `data/sft/combined`, chạy A–E–G, đo trên `labels_v1`, viết kết luận.

---

## 4. Đã có trên Curated

143 file; Slither ra JSON được 141 (hai file crash analyzer, không bịa alert). 343 alert trong thang R/A/C.

| Họ | keep | drop | unknown |
| --- | ---: | ---: | ---: |
| R | 36 | 123 | 8 |
| A | 14 | 53 | 11 |
| C | 61 | 13 | 24 |

Họ A keep mỏng vì nhiều lỗ access-control Curated (sai tên constructor, ghi mapping) **không** trùng detector Slither A — đúng ô ABSENT, không phải “sót keep”. Ghi rõ khi viết báo. Không dùng dataset HuggingFace gắn nhãn theo chính Slither (học vẹt tool đang được triage).

Tập học phụ SolidiFI (200 file RAC, bug bơm, **không** vào F1 chính): 6955 alert. Keep/drop R 1385/303, A 1933/2403, C 923/8. SFT train gộp (`data/sft/combined`): R=1847, A=4403, C=1005, J=633. C drop rất mỏng — LoRA-C sẽ lệch keep.

---

## 5. Bước tiếp theo

Làm theo thứ tự; bước sau cần output bước trước.

**Bước 1 — Học bốn adapter (đang chờ)**  
Học R, rồi A, rồi C, rồi J trên `data/sft/combined` (Curated + SolidiFI phụ). F1 / eval sau train chỉ trên `labels_v1`. Cùng một model nền, mỗi vai một LoRA. Xong bước này mới có hệ E.

Kaggle: GPU T4 + Internet, clone git (không zip):

`git clone https://github.com/kham3465/Qu-n-Tr-An-Ninh-M-ng.git`  
rồi `notebooks/kaggle_train.ipynb` hoặc `python scripts/kaggle_train.py`. `--resume` mặc định.

**Bước 2 — Chạy đủ sáu cấu hình trên cùng tập alert**  
A (không model), B, C, D, E, G. Mỗi cấu hình một file dự đoán: `alert_id`, `source`, `decision`. Judge ghi `invented_count`.

**Bước 3 — Đo + biểu đồ**  
Một lệnh sau khi có adapter (và pred nếu đã chạy hội đồng):

`python scripts/train_lora.py --all --do-train` (mặc định `--resume`: bỏ vai đã xong, nối checkpoint).  
`python scripts/after_train.py` — skip `pred_*.jsonl` đã có; `--force` để chạy lại.  
Có GPU và muốn tự chạy B–G + ablation: thêm `--run-hf`.

Ra `reports/BAO-CAO-CHI-SO.md`, `paper_stats.json`, `reports/figures/fig01…fig12.png`: nhãn, SFT, coverage, P/R/F1, F1 theo họ, invent, gain E vs B, ablation tắt LoRA-R, loss train, TP/FP/FN. F1 chính chỉ trên `labels_v1`.

**Bước 4 — Chỉ khi 1–3 xong**  
Thêm tập ngoài Curated (110 ca SmartAudit, Web3Bugs biên dịch được) nếu đủ mẫu; dưới 20 file thì ghi “chưa đủ”, không ép RQ. Viết báo đúng khung *triage cảnh báo*, không viết thành “hệ audit đa tác nhân như SmartAudit”.

Câu bảo vệ ngắn: *Ba họ vì protocol (tín hiệu + nhãn + dị thể FP + mẫu) trên Slither×Curated chỉ R/A/C đủ. Keep/drop là phán trên alert, không trên detector. Sót được kê trong ô ABSENT, không bịa lỗ.*
