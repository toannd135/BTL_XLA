# Kết quả model khung xoay `obb_colab`

Trọng số: `/content/drive/MyDrive/fruit_runs/obb/obb_colab/weights/best.pt`

## 5.1 Detector (test 373 ảnh, conf 0.25, NMS IoU 0.7)

|  | A | B | C (đang dùng) | OBB `obb_colab` |
|---|---|---|---|---|
| mAP50 | 0.903 | 0.910 | 0.889 | 0.817 |
| mAP50-95 | 0.758 | 0.759 | 0.732 | 0.658 |
| AP50 chuối | 0.802 | 0.836 | 0.805 | 0.637 |
| AP50 táo | 0.927 | 0.935 | 0.905 | 0.844 |
| AP50 cam | 0.979 | 0.960 | 0.958 | 0.970 |
| Ảnh nền báo sai (trên 71 ảnh giữ riêng) | - | 34 (48%) | 3 (4%) | 4 (6%) |

A/B/C: số trong README (test box thẳng). OBB: cùng 373 ảnh, nhãn chuối một phần đã đổi sang khung xoay -> so tương đối.

## 5.2 Ngưỡng conf 0.40 (NMS 0.6, như backend)

| Chỉ số | C (README) | OBB `obb_colab` |
|---|---|---|
| Precision / Recall / F1 tổng | 0.80 / 0.84 / 0.82 | 0.78 / 0.76 / 0.77 |
| F1 chuối (P / R) | 0.73 | 0.65 (0.71 / 0.60) |
| F1 táo (P / R) | 0.89 | 0.86 (0.87 / 0.85) |
| F1 cam (P / R) | 0.90 | 0.85 (0.79 / 0.93) |
| Ảnh nền báo sai (conf 0.40) | 1/71 | 3/71 |
| Độ trễ detector (trung vị, ms/ảnh, GPU Colab) | 16.5 (bản B, máy local) | 7.6 |

Ghép box theo probIoU >= 0.5 (cách Ultralytics đo OBB). TP/FP/FN tổng: 265/75/84.

## Đếm quả (conf 0.40, NMS 0.6)

| Tập / chỉ số | C (box thẳng) | OBB `obb_colab` |
|---|---|---|
| naichuoi.jpeg: số chuối (thật ~20) | 12 | 12 |
| test: MAE chuối / đếm đúng | 0.434 / 0.698 | 0.708 / 0.491 |
| test: MAE táo / đếm đúng | 0.182 / 0.852 | 0.25 / 0.852 |
| test: MAE cam / đếm đúng | 0.143 / 0.893 | 0.155 / 0.857 |
| synth_holdout: MAE chuối / đếm đúng | 6.139 / 0.026 | 0.63 / 0.57 |
| synth_holdout: MAE táo / đếm đúng | 8.473 / 0.0 | 0.275 / 0.769 |
| synth_holdout: MAE cam / đếm đúng | 7.112 / 0.025 | 0.275 / 0.75 |
| neg: ảnh báo sai (conf 0.4) | 1 | 3 |

test: nhãn gốc còn box nải nên chỉ so tương đối; synth_holdout: ảnh ghép, lạc quan.
