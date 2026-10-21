# Locked sg12_crf results

VOC only selects. SG15 source thresholds selected on four training folds; refined outputs inherit them. Full-VOC choices frozen before COCO metrics are interpreted. Diagnostic variants never promoted on COCO.

Same-generation audited controls; historical cross-hardware discrepancies are diagnostic only. Does not certify historical bit-identical reproduction.

| Cell | VOC OOF mIoU | Full VOC threshold | Full VOC mIoU | COCO mIoU |
|---|---:|---:|---:|---:|
| W/REF | 0.787280 | 0.425 | 0.788610 | 0.617163 |
| W/P1 | 0.799652 | 0.450 | 0.800585 | 0.624471 |
| W/SG12/top50 | 0.796658 | 0.450 | 0.798558 | 0.627174 |
| W/P1+R2 | 0.813209 | 0.450 | 0.813209 | 0.637141 |
| W/SG12/top50+R2 | 0.811526 | 0.425 | 0.813126 | 0.631503 |

## Protocol selection

| Screen | Winner | Votes | Selected OOF / REF | Gain | Frozen |
|---|---|---:|---:|---:|---|
| primary | W/P1+R2 | 5/5 | 0.813209 / 0.813209 | +0.000000 | W/P1+R2 |

Forced candidate OOF and protocol-selected OOF are distinct. All five fold winners, thresholds, class IoUs, deltas, paired 20k-bootstrap intervals and improved/worsened/tied counts are in summary.json and fold_results.json.

## Mechanism diagnostics

```json
{
  "voc": {
    "same_input_threshold": 0.425,
    "FN_to_TP": 232075,
    "FP_to_TN": 1325589,
    "TP_to_FN": 381867,
    "TN_to_FP": 273015,
    "new_fp_attached": 248469,
    "new_fp_separate_component": 24546,
    "new_fp_more_than_16px_from_GT": 147969,
    "new_fp_more_than_64px_from_GT": 41057,
    "boundary_error_pixels_before_after": [
      2749953,
      2154661
    ],
    "nonboundary_error_pixels_before_after": [
      8646874,
      8339384
    ],
    "raw_to_crf_same_threshold": {
      "candidate_miou": 0.813125829808936,
      "reference_miou": 0.7970952081888916,
      "delta": 0.016030621620044427,
      "paired_mean_image_iou": {
        "mean": 0.020684050856587345,
        "ci_low": 0.01720006655176491,
        "ci_high": 0.024336899658527703,
        "better": 370,
        "worse": 120,
        "ties": 0
      }
    },
    "raw_to_crf_individually_VOC_selected_thresholds": {
      "candidate_miou": 0.813125829808936,
      "reference_miou": 0.7985584168680424,
      "delta": 0.014567412940893587,
      "paired_mean_image_iou": {
        "mean": 0.017473596166512857,
        "ci_low": 0.01355229049016926,
        "ci_high": 0.021464842765939244,
        "better": 328,
        "worse": 162,
        "ties": 0
      }
    },
    "runtime": {
      "images": 490,
      "shared_generation_and_seed_mean_seconds": [
        6.153692025524013,
        0.028407669189024945
      ],
      "ordinary_two_pass_mean_seconds": 0.16564829203547263,
      "SG12_observation_propagation_mean_seconds": 0.18193638054083805,
      "SG12_ranking_mean_seconds": 0.001133741134283494,
      "SG12_selected_average_mean_seconds": 2.815702131816319e-05,
      "crf_reference_mean_seconds": 0.3012611012677757,
      "crf_candidate_mean_seconds": 0.3013776877704932,
      "reference_inclusive_per_image_latency_mean_seconds": 6.649009088016286,
      "candidate_inclusive_per_image_latency_mean_seconds": 6.666575661179971,
      "measurement_scope": "synchronized isolated operators plus shared live generation/CA seed and one selected-threshold CRF; excludes model loading/metric/report overhead; shared generation is charged in full to each. Not independent end-to-end job benchmarks: no speedup claim."
    }
  },
  "coco": {
    "same_input_threshold": 0.425,
    "FN_to_TP": 185784,
    "FP_to_TN": 2256274,
    "TP_to_FN": 478735,
    "TN_to_FP": 508152,
    "new_fp_attached": 440087,
    "new_fp_separate_component": 68065,
    "new_fp_more_than_16px_from_GT": 388105,
    "new_fp_more_than_64px_from_GT": 203404,
    "boundary_error_pixels_before_after": [
      3147160,
      2446771
    ],
    "nonboundary_error_pixels_before_after": [
      22389860,
      21635078
    ],
    "raw_to_crf_same_threshold": {
      "candidate_miou": 0.6315025777228536,
      "reference_miou": 0.6157494521600931,
      "delta": 0.01575312556276043,
      "paired_mean_image_iou": {
        "mean": 0.03077881127759898,
        "ci_low": 0.02669598791514564,
        "ci_high": 0.03513470956781305,
        "better": 549,
        "worse": 157,
        "ties": 1
      }
    },
    "raw_to_crf_individually_VOC_selected_thresholds": {
      "candidate_miou": 0.6315025777228536,
      "reference_miou": 0.6271735100277616,
      "delta": 0.004329067695092004,
      "paired_mean_image_iou": {
        "mean": 0.01826150397839437,
        "ci_low": 0.014190224786513913,
        "ci_high": 0.022462361556093397,
        "better": 407,
        "worse": 299,
        "ties": 1
      }
    },
    "runtime": {
      "images": 707,
      "shared_generation_and_seed_mean_seconds": [
        6.146080932088586,
        0.028839784240958715
      ],
      "ordinary_two_pass_mean_seconds": 0.16645389038653152,
      "SG12_observation_propagation_mean_seconds": 0.18227401774118443,
      "SG12_ranking_mean_seconds": 0.000897672591148711,
      "SG12_selected_average_mean_seconds": 2.8327592191466852e-05,
      "crf_reference_mean_seconds": 0.31243443482751726,
      "crf_candidate_mean_seconds": 0.3125595533232372,
      "reference_inclusive_per_image_latency_mean_seconds": 6.653809041543593,
      "candidate_inclusive_per_image_latency_mean_seconds": 6.670680287577307,
      "measurement_scope": "synchronized isolated operators plus shared live generation/CA seed and one selected-threshold CRF; excludes model loading/metric/report overhead; shared generation is charged in full to each. Not independent end-to-end job benchmarks: no speedup claim."
    }
  }
}
```
