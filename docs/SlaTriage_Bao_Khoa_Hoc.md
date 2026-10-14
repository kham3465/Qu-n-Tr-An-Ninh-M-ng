# SlaTriage: Adjudicating Slither Alerts with a Family-Specialized LoRA Council

**SlaTriage: phân xử cảnh báo Slither bằng hội đồng LoRA theo họ lỗ**

Nguyễn Gia Khảm, Nguyễn Thị Phương Liên, Nguyễn Thị Len  
Nhóm 19 — Học phần Quản trị An ninh mạng, Đại học Bách khoa Hà Nội  
Tháng 10 năm 2026

---

## Tóm tắt

Công cụ phân tích tĩnh như Slither sinh nhiều cảnh báo trên hợp đồng Solidity, trong đó tỷ lệ dương tính giả cao. Một mô hình ngôn ngữ lớn (LLM) nếu được giao nhiệm vụ “tìm lỗ từ trang trắng” thì dễ bịa detector hoặc số dòng không có trong mã. Bài báo đặt bài toán **adjudication**: với mỗi cảnh báo Slither đã có, quyết định *keep* (giữ) hoặc *drop* (bỏ). Chúng tôi đề xuất **SlaTriage**, hội đồng bốn vai gồm ba chuyên gia theo họ lỗ reentrancy / access-control / unchecked-call (R/A/C) và một Judge cấm invent. Mỗi vai là một adapter QLoRA trên Qwen2.5-Coder-7B. F1 chính chỉ tính trên 34 file hold-out (108 alert keep/drop) không nằm trong tập học. Trên hold-out, cấu hình đầy đủ E đạt **F1 = 0,861** (P = 0,838, R = 0,886), cao hơn baseline Slither thô A (0,490), LLM một prompt B (0,515), hội đồng chưa học C (0,613) và ba LoRA chưa học Judge D (0,757). Ablation tắt chuyên gia R làm F1 hold-out còn 0,831 và F1 toàn tập tụt từ 0,882 xuống 0,786. Bài không nhận SOTA, không nhận “không sót mọi lỗ”, và không đưa cấu hình 14B (G) vào bảng vì gần như toàn bộ đầu ra không parse được JSON.

**Từ khóa:** triage cảnh báo, Slither, LoRA, hợp đồng thông minh, dương tính giả, SmartBugs Curated.

## Abstract

Static analyzers such as Slither emit many Solidity warnings, most of which are false positives. A large language model asked to *discover* bugs from source easily invents detectors or line numbers. We study **alert adjudication**: for each existing Slither finding, decide *keep* or *drop*. **SlaTriage** is a four-role council—three family experts (reentrancy / access-control / unchecked-call) plus a Judge that forbids invented findings—each implemented as a QLoRA adapter on Qwen2.5-Coder-7B. Primary F1 is reported only on a 34-file hold-out (108 labeled alerts) excluded from all supervised data. Full council E reaches **F1 = 0.861** on hold-out, above raw Slither (0.490), a single-prompt 7B (0.515), a prompt-only council (0.613), and LoRA experts with an untrained Judge (0.757). Removing the reentrancy expert drops hold-out F1 to 0.831 and full-set F1 from 0.882 to 0.786. We do not claim SOTA or zero misses. A 14B single-prompt run is omitted from the F1 table because almost every generation failed to parse as JSON.

**Keywords:** alert triage, Slither, LoRA, smart contracts, false positives, SmartBugs Curated.

---

## 1. Giới thiệu

Hợp đồng thông minh sau khi triển khai gần như không sửa được; một lỗ reentrancy hoặc unchecked call có thể rút quỹ. Slither [1] quét nhanh theo hàng chục detector, nhưng auditor phải đọc khối cảnh báo ồn. Hai sai lầm đối xứng: tin hết tool thì lãng phí công; giao cho LLM “audit như người” thì model hay **invent** — bịa loại lỗ hoặc dòng không tồn tại.

Bài toán thực tế của quản trị mã không phải viết detector mới, cũng không phải săn lỗ từ trang trắng theo kiểu hội đồng chức danh [2]. Việc cần làm là **phân xử** từng cảnh báo đã có: keep hoặc drop. Đơn vị quyết định là *alert* (một lần detector bắn trên một file/dòng), không phải tên detector.

**Đóng góp.** (i) Protocol lập họ lỗ: một họ chỉ được mở khi có tín hiệu máy, nhãn người, câu hỏi dương tính giả riêng, và đủ mẫu; trên Slither × SmartBugs Curated [3] chỉ còn ba họ R/A/C. (ii) Hội đồng QLoRA bốn vai (ba chuyên gia + Judge chống invent). (iii) Đo F1 trên hold-out file-level, phạt alert thiếu dự đoán như drop, tách `parse_fail`, kèm khoảng tin cậy bootstrap 95%. Số liệu trong bài lấy từ lần chạy đã lưu, không nội suy.

Chúng tôi **không** nhận phủ mọi lỗ trên Curated. Durieux et al. [4] cho thấy gộp nhiều công cụ tĩnh cũng chỉ đạt khoảng 42% lỗ đã gắn nhãn; lỗ không có alert máy được ghi ABSENT, không bịa finding.

## 2. Công trình liên quan

**Phân tích tĩnh.** Slither [1] là công cụ đồ thị trung gian phổ biến. SmartBugs Curated [3] gắn nhãn người theo hạng DASP. Bài này dùng Slither làm *bus* cảnh báo, không thay detector.

**LLM cho lỗ hợp đồng.** LLM-SmartAudit [2] chia vai theo chức danh và sinh lỗ từ mã. GPTLens [5] nhân bản cùng một prompt. GPTScan [6] xác nhận lỗ logic. iAudit [7] tinh chỉnh trên hàm. SlaTriage khác: vai = họ lỗ; nhãn = keep/drop trên alert; cấm invent dòng/detector.

**Tinh chỉnh hiệu quả.** LoRA [8] và QLoRA [9] cho phép học adapter hạng thấp trên GPU 16 GB. Chúng tôi học bốn adapter trên một nền 7B 4-bit, suy luận bằng cách đổi adapter chứ không load bốn bản mô hình.

## 3. Phát biểu bài toán

Cho tập cảnh báo Slither $\mathcal{A}$ đã xếp họ $f(a)\in\{R,A,C\}$ theo bảng detector. Mỗi $a$ có nhãn vàng $y\in\{\mathrm{keep},\mathrm{drop},\mathrm{unknown}\}$. Mục tiêu học ánh xạ $\hat y(a)\in\{\mathrm{keep},\mathrm{drop}\}$ trên các $a$ có $y\neq\mathrm{unknown}$. Detector ngoài bảng bị loại khỏi thang chính. Khớp nhãn: cùng họ và $|\ell_{\mathrm{alert}}-\ell_{\mathrm{gold}}|\le 5$ dòng thì keep; không khớp thì drop; thiếu dòng hoặc đa khớp thì unknown (không học, không tính F1).

**Invent.** Finding do Judge (hoặc hội đồng) đưa ra mà không nằm trong tập keep của chuyên gia, hoặc số dòng không có trong file nguồn. Đo bằng `invented_count`.

**Không sót quyết định.** Mọi alert in-scope phải có quyết định. Alert vàng không có dự đoán được tính như drop và đếm `n_missing` [mục 5]. Lỗ không có alert máy $\neq$ “sót keep”: đó là ô ABSENT trên ma trận (hợp đồng × hạng DASP).

## 4. Phương pháp

### 4.1. Ba họ lỗ

Một họ chỉ lập khi đủ bốn điều: Slither sinh alert; Curated có keep và drop cùng loại; câu hỏi FP khác họ đã có; đủ mẫu sau khi loại unknown. Kết quả protocol trên bằng chứng hiện có là $k=3$:

| Họ | Câu hỏi | Detector điển hình |
| --- | --- | --- |
| R | Gọi ngoài / chuyển ether trước khi cập nhật state — lỗ hay benign? | `reentrancy-*`, `eth-send` |
| A | Quyền gọi / xác thực / gửi ether có bị kiểm sai? | `tx-origin`, `suicidal`, `arbitrary-send-eth` |
| C | Bỏ qua giá trị trả về hoặc `delegatecall` nguy hiểm? | `unchecked-lowlevel`, `unchecked-send` |

`timestamp` / `tautology` / `unused-return` để thang phụ O. Informational (`solc-version`, …) loại khỏi thang. LLM không tự chọn họ: `assign_family()` tra `detectors_map.yaml`.

Judge không phải họ thứ tư: J chỉ giữ alert chuyên gia đã keep.

### 4.2. Hội đồng và các cấu hình

Một process HuggingFace tải một Qwen2.5-Coder-7B-Instruct 4-bit NF4 và đổi adapter. Nhiệt độ 0.

| ID | Hệ | Mục đích so sánh |
| --- | --- | --- |
| A | Mọi alert in-scope = keep | Trần tool thô |
| B | Một 7B, một prompt, không LoRA | Mốc LLM đơn |
| C | Bốn vai, chỉ prompt | Chia vai chưa học |
| D | 3 LoRA + Judge chưa học | Học chuyên gia |
| E | 3 LoRA + Judge đã học | Hệ đề xuất |
| E\_noR | E tắt chuyên gia R | Ablation họ R |

Cấu hình G (DeepSeek-R1-Distill-Qwen-14B, một prompt) đã chạy trên 112 file nhưng `parse_fail` gần toàn bộ (chuỗi `<think>` chiếm budget sinh JSON). G **không** vào bảng F1.

### 4.3. Học QLoRA

Bốn adapter $r=16$, 4-bit, AMP tắt trên GPU T4 (tránh BF16). SFT gộp Curated (đã loại hold-out) và SolidiFI chỉ như tập phụ. Mẫu Judge dùng đúng prompt lúc chạy: đầu vào là quyết định chuyên gia *giả lập* (nhãn thật, lật ngẫu nhiên 10%, seed cố định), đích là các alert đã keep, `invented_count=0` — Judge không thấy nhãn vàng dưới dạng đáp án.

SFT train: R = 1785, A = 4380, C = 982, J = 1323. SFT Curated (sau hold-out): R = 97, A = 44, C = 51, J = 84. Họ C drop rất mỏng — đây là hạn chế đã biết, không giấu.

### 4.4. Chống rò rỉ và metric

34/112 file (seed 0, 30%) nằm trong `holdout_sources.json`, **loại khỏi mọi SFT**. Hội đồng suy luận trên cả 112 file (343 alert) để có pred đủ; **F1 chính chỉ trên hold-out** (108 alert keep/drop). `score_aligned` đi theo vàng: thiếu pred = drop + `n_missing`. JSON hỏng đếm `parse_fail`, không được ngụy trang thành drop im lặng. F1 kèm CI 95% bootstrap (1000 lần).

## 5. Thiết lập thực nghiệm

**Gold.** SmartBugs Curated: 143 file, Slither ra JSON được 141. Thang R/A/C: 343 alert (300 keep/drop, 43 unknown). Phân bố Curated: R 36/123/8, A 14/53/11, C 61/13/24 (keep/drop/unknown). SolidiFI: 6955 alert RAC, không vào F1.

**Hold-out.** 34 file, 108 alert chấm điểm. Họ A và C trên hold-out chỉ 23 alert mỗi họ — F1 theo họ phải đọc kèm CI.

**Phần cứng.** Học và suy luận trên GPU T4 (Kaggle/Colab), một GPU. Máy CPU chỉ dựng báo cáo và figure sau khi có `pred_*.jsonl`.

**Figure.** Bộ 18 hình nằm tại `slatriage/reports/figures/` (nhãn, SFT, coverage, P/R/F1, invent, gain, ablation, loss, nhầm lẫn).

## 6. Kết quả

Bảng 1 là F1 **chính** (hold-out). Bảng 2 là toàn tập 300 keep/drop — có file đã thấy lúc học, chỉ để tham khảo, không dùng làm kết luận generalization.

**Bảng 1.** Hold-out (34 file, $n=108$). CI 95% bootstrap trên F1.

| Config | P | R | F1 | CI F1 | TP / FP / FN | invent |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| A | 0,324 | 1,000 | 0,490 | [0,39–0,58] | 35 / 73 / 0 | 0 |
| B | 0,394 | 0,743 | 0,515 | [0,40–0,63] | 26 / 40 / 9 | 0 |
| C | 0,447 | 0,971 | 0,613 | [0,51–0,70] | 34 / 42 / 1 | 1 |
| D | 0,718 | 0,800 | 0,757 | [0,64–0,85] | 28 / 11 / 7 | 75 |
| **E** | **0,838** | **0,886** | **0,861** | **[0,76–0,93]** | **31 / 6 / 4** | 50 |
| E\_noR | 0,762 | 0,914 | 0,831 | [0,73–0,91] | 32 / 10 / 3 | 60 |

**Bảng 2.** Toàn `labels_v1` ($n=300$, không phải F1 chính).

| Config | P | R | F1 | CI F1 |
| --- | ---: | ---: | ---: | --- |
| A | 0,370 | 1,000 | 0,540 | [0,48–0,59] |
| B | 0,418 | 0,847 | 0,560 | [0,50–0,62] |
| C | 0,460 | 0,991 | 0,629 | [0,57–0,68] |
| D | 0,744 | 0,865 | 0,800 | [0,74–0,85] |
| **E** | **0,890** | **0,874** | **0,882** | **[0,83–0,93]** |
| E\_noR | 0,692 | 0,910 | 0,786 | [0,73–0,84] |

**So mốc.** E hơn B trên hold-out: F1 0,515 → 0,861 (gain tương đối khoảng +67%). C hơn B (0,613 vs 0,515): chia vai prompt đã có ích nhưng còn xa E. D đã kéo mạnh (0,757) nhờ LoRA chuyên gia, song Judge chưa học bịa 75 finding; E giảm invent còn 50 và tăng F1. Ablation: tắt R, hold-out 0,831 (vẫn dưới E), full-set 0,882 → 0,786 — chuyên gia R không dư.

**Theo họ (hold-out, E).** R: F1 0,857 ($n=62$); A: 0,833 ($n=23$); C: 0,872 ($n=23$). Họ A/C mẫu mỏng, CI rộng (A tới [0,55–1,00]).

**Coverage DASP (Curated, 143 hợp đồng).** Reentrancy: KEEP 35, DROP 24, ABSENT 83. Unchecked low-level: KEEP 52, DROP 5, ABSENT 82. Access-control: KEEP 12, DROP 46, ABSENT 78. Arithmetic, front-running, bad randomness, DoS, time manipulation: ABSENT 143 — không có detector Slither tương ứng, không tính là “đã bắt”.

A đạt R = 1 vì mọi alert in-scope đều bị gán keep: precision thấp (0,32 hold-out) đúng bản chất tool ồn.

## 7. Thảo luận và hạn chế

Hold-out 108 alert là nhỏ; F1 họ A/C dao động — phải đọc CI, không được viết thành “họ A đã giải quyết”. LoRA-C lệch keep vì thiếu drop (Curated còn ít drop C; SolidiFI C drop = 8). SolidiFI là bug bơm, phân bố khác Curated, chỉ được dùng để học phụ. Mẫu Judge dùng chuyên gia giả lập, chưa phải chuỗi R/A/C thật lúc suy luận. G 14B không vào bảng: đây là thất bại kỹ thuật (format), không phải bằng chứng “7B hơn 14B”.

Chúng tôi không so GPT thương mại, không nhận vượt trần phủ lỗ của phân tích tĩnh [4], không nhận zero miss.

## 8. Kết luận

SlaTriage biến Slither từ danh sách ồn thành quyết định keep/drop theo từng alert. Ba họ là kết quả protocol trên Slither × Curated, không phải tiên đề. Hội đồng QLoRA đầy đủ (E) đạt F1 hold-out 0,861, hơn rõ mốc tool thô, LLM một prompt, và hội đồng chưa học; Judge đã học giảm invent so với D; tắt R làm giảm F1. Hướng mở: mẫu drop cho họ C, Judge trên chuỗi chuyên gia thật, tập ngoài Curated khi đủ file biên dịch, và (nếu cần) chạy lại baseline 14B với ngân sách sinh đủ để ra JSON sau `</think>`.

---

## Tài liệu tham khảo

[1] J. Feist, G. Grieco, and A. Groce, “Slither: A static analysis framework for smart contracts,” in *Proc. IEEE/ACM WETSEB*, 2019.

[2] Z. Wei, J. Sun, Z. Zhang, X. Zhang, M. Li, and Z. Li, “LLM-SmartAudit: Advanced smart contract vulnerability detection,” *arXiv* / công trình hội đồng đa vai trên mã nguồn, 2024.

[3] J. F. Ferreira, P. Cruz, T. Durieux, and R. Abreu, “SmartBugs: A framework to analyze Solidity smart contracts,” in *Proc. ASE*, 2020. Tập Curated: nhãn người theo hạng DASP.

[4] T. Durieux, J. F. Ferreira, R. Abreu, and P. Cruz, “Empirical review of automated analysis tools on 47,587 Ethereum smart contracts,” in *Proc. ICSE*, 2020.

[5] S. Hu, T. Huang, F. İlhan, S. F. Tekin, and L. Liu, “Large language model-powered smart contract vulnerability detection: New perspectives” (GPTLens), in *Proc. IEEE TPS-ISA*, 2023.

[6] Y. Sun, D. Wu, Y. Xue, H. Liu, H. Wang, Z. Xu, X. Xie, and Y. Liu, “GPTScan: Detecting logic vulnerabilities in smart contracts by combining GPT with program analysis,” in *Proc. ICSE*, 2024.

[7] W. Ma, D. Wu, Y. Sun, T. Wang, S. Liu, J. Zhang, Y. Xue, and Y. Liu, “Combining fine-tuning and LLM-based agents for intuitive smart contract auditing” (iAudit), *arXiv:2403.16073*, 2024.

[8] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022.

[9] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in *Proc. NeurIPS*, 2023.

[10] DASP Top 10, NCC Group, “Decentralized Application Security Project,” 2018. https://dasp.co/

---

## Phụ lục: nguồn số và tái lập

- Gold: `slatriage/data/labels/labels_v1.jsonl` (343 dòng, 300 keep/drop).
- Hold-out: `slatriage/data/labels/holdout_sources.json` (34 file, seed 0).
- Pred: `Downloads/report_BCDE/pred_{A,B,C,D,E,E_noR}.jsonl` (A/E từ lần chạy trước, cùng 343/112).
- Metric: `score_aligned` trong `slatriage/src/slatriage/metrics.py`; bảng tổng hợp `paper_stats.json`, `BAO-CAO-CHI-SO.md`.
- Figure: `slatriage/reports/figures/fig01`–`fig18.png`.
- Mã: https://github.com/kham3465/Qu-n-Tr-An-Ninh-M-ng
- G không dùng cho F1 (`parse_fail` ≈ 339/339 trên bản chạy 14B).
