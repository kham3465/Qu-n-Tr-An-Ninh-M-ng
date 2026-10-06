# Outline bài báo hội nghị trong nước — SlaTriage

**Độ dài mục tiêu:** 6–8 trang (theo template CFP).  
**Ngôn ngữ:** ưu tiên tiếng Việt.  
**Framing:** phân xử (triage) cảnh báo Slither — không phải audit đa tác nhân kiểu LLM-SmartAudit.

---

## Tiêu đề (chọn 1)

- Tiếng Việt: *SlaTriage: Hội đồng đa tác nhân LoRA phân xử cảnh báo Slither trên hợp đồng thông minh*  
- English: *SlaTriage: Multi-Agent LoRA Adjudication of Slither Alerts for Smart Contracts*

**Từ khóa:** hợp đồng thông minh; Slither; đa tác nhân; LoRA; cảnh báo giả; LLM  

---

## 1. Mở đầu (~1 trang) — P3 nháp

Đoạn 1: Smart contract + rủi ro; công cụ tĩnh (Slither) **ồn**.  
Đoạn 2: LLM hỗ trợ audit nhưng **ảo giác / invent**. Multi-agent chức danh đã có (LLM-SmartAudit).  
Đoạn 3: Gap — thiếu hội đồng **train theo họ lỗ**, đầu vào là **alert tool**, đo **invent**.  
Đoạn 4: Đề xuất SlaTriage (R/A/C/J, Qwen-7B LoRA, Judge cấm invent).  
Đoạn 5: **Ba đóng góp C1–C3** (xem `PHAN-VIEC-3-NGUOI-HOI-NGHI-VN.md` §0.3).  
Đoạn 6: Tóm tắt kết quả chính (điền số sau thí nghiệm; nếu không đạt 15% vẫn nêu trung thực).

---

## 2. Cơ sở lý thuyết / Related Work (~1 trang) — P3

Bảng so sánh cột: Đầu vào | Học? | Multi-agent kiểu gì | Đo invent?

Hàng: Slither; LLM-SmartAudit; GPTScan; GPTLens; iAudit; MOS; **SlaTriage (ours)**.

Mỗi hệ 3–5 câu. Kết đoạn: khoảng trống adjudication alert + invent.

---

## 3. Phương pháp (~1.5–2 trang) — P3 biên / P2 bổ sung train

### 3.1. Tổng quan pipeline  
Hình: `.sol → solc → Slither → map R/A/C → LoRA experts → Judge → JSON`

### 3.2. Chuyên gia R, A, C  
Phạm vi họ; cấm bắt ngoài họ; input/output JSON keep/drop.

### 3.3. Judge và ràng buộc invent  
Định nghĩa invent; SFT/DPO.

### 3.4. Huấn luyện QLoRA (P2 viết số liệu)  
Base model; hyperparam; không pretrain lại.

### 3.5. Cấu hình so sánh A–E–G  
Bảng ký hiệu như Week 3.

---

## 4. Dữ liệu và thiết lập thí nghiệm (~1 trang) — P1 số + P3 viết

### 4.1. SmartBugs Curated  
### 4.2. 110 ca + nhóm sạch  
### 4.3. Quy tắc khớp keep/drop/unknown + chấm tay ≥100  
### 4.4. Web3Bugs (hoặc hạn chế mẫu)  
### 4.5. Metrics: P/R/F1, gain tương đối, % invent  
### 4.6. Chi tiết triển khai: GPU, VRAM, temperature 0  

---

## 5. Kết quả và thảo luận (~1.5–2 trang) — P3

### 5.1. RQ1 (E vs B trên trung gian)  
### 5.2. RQ2 (Web3Bugs / chưa đủ mẫu)  
### 5.3. RQ3 (invent C/D/E)  
### 5.4. Ablation tắt LoRA  
### 5.5. So với G (DeepSeek-14B)  
### 5.6. Phân tích lỗi / case định tính  
### 5.7. Hạn chế  

---

## 6. Kết luận (~0.5 trang) — P3

Tóm C1–C3; hướng mở (thêm họ lỗ, time-split, DPO Judge).

---

## Phụ lục (nếu CFP cho phép)

Lệnh reproduce ngắn; ví dụ JSON alert/decision.

---

## Phân đoạn viết (deadline nội bộ)

| Mục | Chịu trách nhiệm nội dung | Deadline gợi ý |
| --- | --- | --- |
| §1, §2, §5, §6 | P3 | Tuần 7–8 |
| §3.1–3.3, §3.5 | P3 + review P2 | Tuần 7 |
| §3.4 | P2 | Tuần 7 |
| §4 | P1 số → P3 câu chữ | Tuần 6–7 |
| Hình pipeline | P3 (hoặc P2) | Tuần 6 |
