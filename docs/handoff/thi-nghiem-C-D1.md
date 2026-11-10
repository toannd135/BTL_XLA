# Bàn giao 2026-11-08: detector C xong, D1 (ảnh ghép) đang train

## Mục tiêu
Giảm nhận nhầm vật khác thành quả (C, xong) và nhận đúng cảnh nhiều loại quả, nhất là táo xanh (D1, đang chạy).

## Git
Nhánh master, HEAD 9c08265 "Update retrain handoff". 10 tệp chưa commit (sửa det_neg.py, main.py, App.jsx, CLAUDE.md, inference_config.json, tệp mới). Chưa commit/push gì hôm nay. Người dùng tự gõ lệnh git.

## Đã xong (chưa commit)
- C: fine-tune từ B + 287 ảnh nền x2, dùng last.pt epoch 11. Nền báo sai 34/71 -> 3/71; test mAP50 0.910 -> 0.889, mAP50-95 0.759 -> 0.732; AP50 chuối 0.836 -> 0.805. Đã vào exports/det_finetune_best.pt/.onnx (= det_C_neg.pt). Backup: det_B_clean_backup.*, det_A_baseline_backup.*.
- Ngưỡng 0.40: backend CONF, frontend slider (App.jsx), inference_config.json. Backend chưa restart với C.
- Chẩn đoán ảnh stock chuối+táo xanh+cam: A/B/C đều 0 box táo. Dữ liệu train chỉ 21/2365 ảnh >=2 loại quả.

## Quyết định đã chốt
- Chọn last.pt thay best.pt (val không có ảnh nền). Ngưỡng chung 0.40, không tách theo class.
- own_mixed/ (nhãn nháp YOLO11l) quá bẩn, KHÔNG train. Dùng ảnh ghép tổng hợp thay thế.

## Tệp quan trọng
- work/exp/det_neg.py (C), work/exp/det_synth.py (D1), work/gen_synth.py (ảnh ghép), work/diag_image.py (chẩn đoán ảnh)
- work/exp/neg/test/images: 71 ảnh nền giữ riêng. Đừng chạy lại det_neg.py khi chưa cần.
- CLAUDE.md mục "Kế hoạch tiếp theo" mục 0; backend/main.py:27 (CONF)

## Việc còn lại
1. So C vs D1 (conf 0.25, IoU 0.7): test thật 373 ảnh, 71 ảnh nền, diag_image.py trên ảnh stock (/home/toannd/Desktop/green-apples-oranges-bananas-on-260nw-2671368241.png).
2. Nhận D1 nếu mAP50 test giảm <= ~0.01 so với C, nền <= ~6/71, ra box táo xanh. Nếu nhận: copy last.pt vào exports, xuất onnx, quét lại ngưỡng, cập nhật CLAUDE.md.
3. Cần người: ảnh thật nhiều loại quả làm bộ test độc lập; sửa nhãn ~30 ảnh test bị cắm cờ.

## Đang chờ
Lượt train D1 chạy nền (20 epoch, ~35 phút), trọng số runs/exp/det_D_synth/weights/last.pt. Không chạy GPU khác khi đang train.

## Lệnh kiểm tra
- `pgrep -f "python work/[e]xp/det_synth"` (còn chạy?) và `tail -3 runs/exp/det_D_synth/results.csv`
- `python work/diag_image.py <ảnh>`
