# SlaTriage

Phân xử cảnh báo Slither: mỗi **alert** quyết **keep** (giữ, lỗ thật) hoặc **drop** (bỏ, ồn).

Không viết detector mới. Không săn lỗ từ trang trắng kiểu LLM-SmartAudit. Đầu vào là JSON Slither; đầu ra là quyết định trên từng cảnh báo đã có.

Hội đồng bốn vai: ba chuyên gia theo **họ lỗ** R / A / C và một **Judge** (gộp quyết định, không invent finding).

Mã nguồn: https://github.com/kham3465/Qu-n-Tr-An-Ninh-M-ng.git

---

## Tài nguyên và kết quả thực nghiệm

Kho Git chứa mã huấn luyện và đánh giá (`scripts/`, `notebooks/`), nhãn, mẫu SFT, bản thảo bài báo (`docs/SlaTriage_Bao_Khoa_Hoc.md`), các file dự đoán `pred_*.jsonl`, `reports/paper_stats.json` và bộ hình `reports/figures/`. Trọng số LoRA (định dạng `.safetensors`) và nhật ký hội đồng `*.report.json` không được đưa vào Git do dung lượng.

Tập tin bổ sung được lưu trên Google Drive, thư mục `slatriage_reports`:

https://drive.google.com/drive/folders/1hIDVse448FkuoESlLDdXnZsKCVBgke-v?usp=drive_link

Thư mục này gồm các file `pred_*.jsonl`, `*.report.json` và, khi đã tải lên, gói `slatriage_adapters.zip` (bốn adapter R, A, C, J; khoảng 80 MB). Quyền truy cập: chia sẻ cho thành viên nhóm (xem hoặc chỉnh sửa).

Chỉ số F1 chính được tính trên 34 file hold-out (108 cảnh báo keep/drop), không thuộc tập SFT. Bảng dưới đây lấy từ `reports/BAO-CAO-CHI-SO.md`.

| Cấu hình | Mô tả | F1 hold-out | F1 toàn tập (300) |
| --- | --- | ---: | ---: |
| A | Slither thô, mọi alert in-scope được gán keep | 0,490 | 0,540 |
| B | Qwen2.5-Coder-7B, một prompt, chưa LoRA | 0,515 | 0,560 |
| C | Bốn vai, chỉ prompt | 0,613 | 0,629 |
| D | Ba LoRA chuyên gia, Judge chưa huấn luyện | 0,757 | 0,800 |
| E | Ba LoRA và Judge đã huấn luyện | 0,861 | 0,882 |
| E\_noR | Như E, không dùng chuyên gia R | 0,831 | 0,786 |

Cấu hình G (DeepSeek-R1-Distill-Qwen-14B, một prompt) đã được chạy trên 112 file nhưng tỷ lệ `parse_fail` gần toàn bộ; kết quả này không đưa vào bảng F1.

Huấn luyện lại (GPU T4), nếu cần:

```text
git clone https://github.com/kham3465/Qu-n-Tr-An-Ninh-M-ng.git
cd Qu-n-Tr-An-Ninh-M-ng
pip install -r requirements-train.txt
python scripts/kaggle_train.py
```

Cờ `--resume` bỏ qua vai đã có `adapter_config.json` và `train_meta.json`.

Dựng lại bảng và hình trên CPU khi đã có `reports/pred_*.jsonl`:

```text
pip install -r requirements.txt
python -c "from slatriage.report import write_report; write_report()"
```

Suy luận hội đồng trên GPU: giải nén `slatriage_adapters.zip` vào `adapters/{R,A,C,J}` rồi chạy

`python scripts/after_train.py --run-hf --resume --configs E,B --skip-baseline-a`

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

Bài không khẳng định phủ hết mọi lỗ trên Curated. Durieux et al. (ICSE 2020) cho thấy chín công cụ tĩnh gộp lại chỉ bắt khoảng 42% lỗ đã gắn nhãn trên SmartBugs Curated. Yêu cầu đặt ra là **không sót quyết định** trên mọi alert thuộc thang và mọi hạng DASP đã tuyên bố. Lỗ không có alert máy được ghi ở ô **ABSENT** (sót có sổ), không bổ sung finding.

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
5. Xuất mẫu học R, A, C, J (loại unknown **và loại 34 file hold-out**, xem mục 6).
6. Dựng ma trận coverage từ nhãn.
7. Thêm SolidiFI (bug bơm, chỉ họ R/A/C) làm **tập học phụ** → `labels_solidifi.jsonl` + `data/sft/combined`. F1 chính vẫn Curated.

Bốn adapter đã huấn luyện xong. Các file `pred_A` đến `pred_E` và `pred_E_noR` đủ 343 cảnh báo / 112 file. Bản thảo bài báo nằm tại `docs/SlaTriage_Bao_Khoa_Hoc.md`.

---

## 4. Đã có trên Curated

143 file; Slither ra JSON được 141 (hai file crash analyzer, không bịa alert). 343 alert trong thang R/A/C.

| Họ | keep | drop | unknown |
| --- | ---: | ---: | ---: |
| R | 36 | 123 | 8 |
| A | 14 | 53 | 11 |
| C | 61 | 13 | 24 |

Họ A keep mỏng vì nhiều lỗ access-control Curated (sai tên constructor, ghi mapping) **không** trùng detector Slither A — đúng ô ABSENT, không phải “sót keep”. Ghi rõ khi viết báo. Không dùng dataset HuggingFace gắn nhãn theo chính Slither (học vẹt tool đang được triage).

Tập học phụ SolidiFI (200 file RAC, bug bơm, **không** vào F1 chính): 6955 alert. Keep/drop R 1385/303, A 1933/2403, C 923/8. SFT Curated (đã loại hold-out): R=97, A=44, C=51, J=84. SFT train gộp (`data/sft/combined`): R=1785, A=4380, C=982, J=1323 (mẫu Judge chia lô 6 alert cho khớp lúc chạy thật nên nhiều dòng hơn trước). C drop rất mỏng (còn 7 drop Curated, 15 drop cả bộ gộp, so với 967 keep) — LoRA-C sẽ lệch keep.

---

## 5. Tái lập thực nghiệm

Các lệnh huấn luyện, suy luận và dựng báo cáo được nêu ở mục *Tài nguyên và kết quả thực nghiệm*. Lỗ không có cảnh báo máy được ghi nhận ở ô ABSENT, không được bổ sung finding.

---

## 6. Đánh giá đúng cách (chống rò rỉ)

Bản trước từng chấm F1 trên toàn bộ `labels_v1` trong khi mẫu học cũng lấy từ đó, nên đo ra là khả năng thuộc lòng. Bản hiện tại sửa như sau:

| Vấn đề | Cách xử lý |
| --- | --- |
| Train và test trùng file | `data/labels/holdout_sources.json` (34/112 file, seed 0) bị **loại khỏi mọi SFT**. Hội đồng mặc định chạy **cả 112 file** (343 alert, cùng tập `pred_E`). Báo cáo vẫn **chỉ tính F1 trên hold-out**. `--holdout-only` nếu chỉ muốn 34 file. Colab phần còn lại: `python scripts/colab_eval_rest.py` (B,C,D,E_noR,G; resume từng file; không train lại). |
| Judge học bằng đáp án | Mẫu Judge dùng đúng prompt lúc chạy thật (`judge_system` / `judge_user`): đầu vào là **quyết định chuyên gia giả lập** (nhãn thật, lật ngẫu nhiên 10%, seed cố định), không có nhãn thật. Đích là những alert chuyên gia đã keep, `invented_count=0`. Judge học đúng một luật: chỉ giữ cái chuyên gia giữ, không bịa. Mẫu SolidiFI cũ được chuyển bằng `rewrite_legacy_judge_sft` (không còn nhãn gốc, nên alert thiếu dòng nếu trước đó là drop). |
| Alert bị model bỏ sót | `score_aligned` đi theo **gold**: alert không có dự đoán tính là `drop` và ghi vào `n_missing`. Không còn lợi từ việc im lặng. |
| JSON hỏng bị giấu thành `drop` | Đếm riêng `parse_fail` (trong metric, `*.report.json` và bảng báo cáo). |
| F1 dao động vì mẫu ít | Mỗi F1 kèm khoảng tin cậy 95% bootstrap (`f1_ci`). Họ A, C trên hold-out rất ít alert, nên đọc kèm khoảng này. |
| Prompt dài bị cắt cuối | Cắt từ **bên trái** và in cảnh báo mỗi lần cắt (`HFModelPool.n_truncated`). Danh sách dòng trong prompt Judge giới hạn 4 dòng đầu. |

Còn hạn chế cần ghi vào báo: hold-out chỉ 108 alert keep/drop (họ A, C rất mỏng); LoRA-C vẫn lệch keep vì thiếu drop; SolidiFI là bug bơm nên phân bố khác Curated; mẫu Judge dùng chuyên gia giả lập chứ chưa phải đầu ra của LoRA R/A/C thật.

Kiểm thử: `pip install -r requirements.txt && python -m pytest -q`. `tests/test_leakage.py` giữ các bất biến ở bảng trên (SFT không chứa file hold-out, prompt Judge không chứa nhãn, alert bỏ sót bị phạt).
