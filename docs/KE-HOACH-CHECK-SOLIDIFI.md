# Ke hoach check SolidiFI truoc khi dua vao train

Muc tieu: quyet **co / khong** tron SFT SolidiFI vao LoRA. Curated van la tap F1 chinh.

## Full RAC 200 file — 2026-10-10

Slither **200/200** OK. 6955 alert RAC, `split=solidifi`, cross-family keep = 0. Cong 1–3: **pass**.

| family | keep | drop | unknown | SFT aux |
| --- | ---: | ---: | ---: | ---: |
| R | 1385 | 303 | 0 | 1688 |
| A | 1933 | 2403 | 0 | 4336 |
| C | 923 | 8 | 0 | 931 |
| J | — | — | — | 200 |

SFT train (`data/sft/combined`): R=1847 A=4403 C=1005 J=308. C drop = 8 → LoRA-C lech keep; F1 chinh van Curated.

## Probe 8 file (2 / loai) — 2026-10-09

Slither 8/8 OK. 127 alert RAC, `alert_id` unique. Sau `multi_hit=keep` (cung ho, nhieu vi tri bom):

| family | keep | drop | unknown |
| --- | ---: | ---: | ---: |
| R | 26 | 3 | 0 |
| A | 34 | 47 | 0 |
| C | 17 | 0 | 0 |

Keep A tren **2 file** da 34 (> ca Curated 14). Cong 1–2 tren mau nho: **pass**.

## Da xac minh tren dia (parser)

- 4 thu muc RAC: `Re-entrancy` (1343), `tx.origin` (1336), `Unchecked-Send` (1266), `Unhandled-Exceptions` (1374). Tong **5319** vi tri bom.
- BugLog CSV: `loc`, `length` → day dong. **Khong** dung cot `bug type` (file Re-entrancy bi loi chu).
- Pragma `>=0.4.22 <0.6.0` → solc 0.4.25.
- Script: `python scripts/prepare_solidifi.py --limit-per-type 2`

## 4 cong bat buoc (fail 1 cong = chua merge SFT)

1. **Compile / Slither:** >= 80% file RAC ra JSON (`data/raw/solidifi_slither/summary.json`).
2. **Keep that:** ho A keep >> 14 (ky vong hang tram `tx-origin`). Neu keep A van ~0 → Slither 0.11 khong bat snippet bom, **khong train**.
3. **Khong vo protocol ho:** alert `reentrancy-*` khong keep theo log `tx.origin` (doi ho = drop). Spot-check 20 dong.
4. **Tach tap:** SFT ghi `sft_solidifi_*.jsonl`, label `split=solidifi`. Cam tron vao bang F1 Curated.

## Thu tu chay

```text
A. Probe 2 file / loai
   python scripts/prepare_solidifi.py --limit-per-type 2
   Doc data/labels/stats_solidifi.md
   Cong 1–2 tren mau nho.

B. Neu probe OK: full RAC (200 file, ~20–40 phut)
   python scripts/prepare_solidifi.py --export-sft
   Lai cong 1–3 tren toan tap.
   Script ghi `data/sft/sft_solidifi_*.jsonl` va gop `data/sft/combined/sft_*.jsonl`.
   JSON Slither da co thi skip (them `--force-slither` neu muon chay lai).

C. Train (chi khi B pass)
   Giu 2 bo SFT: `data/sft/sft_*.jsonl` (Curated)  /  `data/sft/combined` (Curated+SolidiFI)
   `train_lora.py` / `kaggle_train.py` uu tien combined.
   F1 chinh: chi labels_v1 Curated
   F1 phu: labels_solidifi (ghi "bug bom")

D. Khong dung
   Overflow, Timestamp, TOD
   slither-audited HF / SmartBugs-Wild
```

## Rui ro can ghi tren bao

- Bug bom, Solc 0.4, snippet lap — model de hoc mau chu `transferTo_txorigin` thay vi phan xu that.
- Unhandled-Exceptions: Slither `unchecked-lowlevel` / `unused-return` co the lech dong.
- Gop vao train chinh lam F1 Curated "dep" gia (phan phoi khac).
