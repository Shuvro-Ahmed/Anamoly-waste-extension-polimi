# Experiment 8 — decoder-only repair training

SMOKE=False. Full dataset run.
Only copied final decoders were optimized. Original encoders, top decoder, pre-quantization layer, codebooks and EMA stayed frozen. No QMem temporal loss was retrained.
Both arms used identical starting weights, image order, epochs, learning rate and optimizer. Ordinary arm: clean-code MSE. Repair arm: normalized clean + 0.5 * repaired-code MSE.
Inference alpha=0.25 fixed from Experiment 7; no alpha or lambda search. Selected epochs: {'ordinary': 1, 'repair': 2}.
Both arms use the same NORMAL-only checkpoint rule. Epoch zero can win; do not describe epoch-zero selection as evidence of beneficial fine-tuning.

## Main comparisons
```
                  stream  pixel_auroc  pixel_ap  historical_iou  primary_iou  normal_audit_fpr  test_normal_pixel_fpr  test_anomaly_pixel_recall  relative_mae_increase  within_budget
christian_reconstruction     0.848375  0.594226        0.337823     0.235794          0.027868               0.007784                   0.242337               1.142232          False
     qmem_reconstruction     0.843143  0.460057        0.355094     0.341938          0.021030               0.070500                   0.488733               0.000000           True
     qmem_bounded_repair     0.849416  0.484371        0.359243     0.350321          0.023048               0.065093                   0.489538               0.024072           True
          ordinary_plain     0.839708  0.440482        0.352574     0.305624          0.021173               0.060874                   0.418903              -0.026024           True
        ordinary_bounded     0.846431  0.467654        0.357506     0.318647          0.022460               0.057541                   0.431361              -0.008278           True
            repair_plain     0.837917  0.436574        0.358097     0.301617          0.021535               0.057595                   0.403204              -0.000922           True
          repair_bounded     0.842243  0.452854        0.361346     0.311608          0.021120               0.057069                   0.415755               0.000086           True
```

## Paired IoU differences at target calibration FPR 0.02
```
           first              second mode  target_fpr         metric  videos     delta  ci95_low  ci95_high
  repair_bounded    ordinary_bounded post        0.02 mean_frame_iou      14 -0.007039 -0.012681   0.000104
  repair_bounded qmem_bounded_repair post        0.02 mean_frame_iou      14 -0.038713 -0.042217  -0.035048
ordinary_bounded qmem_bounded_repair post        0.02 mean_frame_iou      14 -0.031674 -0.036863  -0.027851
    repair_plain      ordinary_plain post        0.02 mean_frame_iou      14 -0.004007 -0.009701   0.002567
  repair_bounded        repair_plain post        0.02 mean_frame_iou      14  0.009991  0.006763   0.013637
```

The causal control is repair_bounded versus ordinary_bounded. The practical control is repair_bounded versus qmem_bounded_repair (the existing frozen method). A gain over original un-repaired QMem alone does not establish value from repair training.
Read achieved audit FPR, AP, and normal damage alongside IoU. Equal calibration targets do not guarantee equal false-alarm rates. Historical all-validation Tukey IoU is a separate secondary protocol.
Checkpoint selection rewards NORMAL reconstruction, not anomaly detection. Training may undo the useful inference effect. Read test_region_score_summary.csv and normal_repair_sensitivity.csv, plus the detection metrics, to distinguish improved normal reconstruction from erased anomaly separation.
Region scores are pixel-weighted, Gaussian-smoothed absolute RGB residuals; changes do not by themselves establish better classification.
This is a one-seed exploratory pilot on repeatedly inspected test videos. Video-bootstrap intervals do not correct experiment selection and do not measure training-seed variability. Ranking intervals concern equal-video metrics, not pooled AP.
Empty-union IoU=1. Audit does not trigger checkpoint reselection. Full three-epoch training budget is matched even when normal selection chooses different epochs.

Share qmem_experiment8_report.zip and the log. Keep qmem_experiment8_decoders.zip for inference with the original QMem checkpoint. No large score arrays are needed for first review.
Decision: retain training only if it adds useful detection performance beyond both controls with acceptable normal damage and false alarms. If ordinary fine-tuning explains the gain, do not attribute it to repair loss. If training fails, retain Experiment 7 and close this training branch.
