# Metrics & output schema (P3 sở hữu)

## Quyết định chuyên gia (R/A/C)

```json
{
  "alert_id": "reentrancy-eth_0",
  "family": "R",
  "decision": "keep",
  "line": 42,
  "reason": "external call before state update"
}
```

`decision`: `keep` | `drop` only.

## Judge

```json
{
  "findings": [
    {"alert_id": "reentrancy-eth_0", "detector": "reentrancy-eth", "line": 42}
  ],
  "invented_count": 0
}
```

**Invent:** finding không có trong danh sách `keep` của chuyên gia, hoặc `line` không tồn tại trong file nguồn.

## Chỉ số

- Precision / Recall / F1 trên nhãn `keep` (bỏ `unknown` khỏi mẫu chấm).
- RQ môn: gain tương đối `(P_E - P_B) / P_B` (mốc kiểm 15%, không ghi title).
- % invent ở C, D, E.

Nhiệt độ inference: **0**.
