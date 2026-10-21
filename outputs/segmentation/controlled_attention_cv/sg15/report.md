# Locked sg15 results

VOC only selects. SG15 source thresholds selected on four training folds; refined outputs inherit them. Full-VOC choices frozen before COCO metrics are interpreted. Diagnostic variants never promoted on COCO.

Same-generation audited controls; historical cross-hardware discrepancies are diagnostic only. Does not certify historical bit-identical reproduction.

| Cell | VOC OOF mIoU | Full VOC threshold | Full VOC mIoU | COCO mIoU |
|---|---:|---:|---:|---:|
| W/REF | 0.787280 | 0.425 | 0.788610 | 0.617163 |
| W/P1 | 0.799652 | 0.450 | 0.800585 | 0.624471 |
| W/SG11/trim20/rho0p80 | 0.798400 | 0.425 | 0.798400 | 0.621916 |
| W/SG12/top50 | 0.796658 | 0.450 | 0.798558 | 0.627174 |
| W/P1/SG15-A025 | 0.771825 | 0.450 | 0.772341 | 0.590097 |
| W/P1/SG15-A050 | 0.785750 | 0.450 | 0.786946 | 0.610955 |
| W/P1/SG15-A075 | 0.781210 | 0.450 | 0.783029 | 0.617070 |
| W/P1/SG15-B | 0.796477 | 0.450 | 0.797453 | 0.622908 |
| W/SG11/trim20/rho0p80/SG15-A025 | 0.774839 | 0.425 | 0.774839 | 0.591763 |
| W/SG11/trim20/rho0p80/SG15-A050 | 0.788200 | 0.425 | 0.788200 | 0.610175 |
| W/SG11/trim20/rho0p80/SG15-A075 | 0.786171 | 0.425 | 0.786171 | 0.613398 |
| W/SG11/trim20/rho0p80/SG15-B | 0.798934 | 0.425 | 0.798934 | 0.622023 |
| W/SG12/top50/SG15-A025 | 0.772646 | 0.450 | 0.774348 | 0.592589 |
| W/SG12/top50/SG15-A050 | 0.783663 | 0.450 | 0.786272 | 0.615600 |
| W/SG12/top50/SG15-A075 | 0.780167 | 0.450 | 0.782186 | 0.618773 |
| W/SG12/top50/SG15-B | 0.794904 | 0.450 | 0.797112 | 0.625986 |

## Protocol selection

| Screen | Winner | Votes | Selected OOF / REF | Gain | Frozen |
|---|---|---:|---:|---:|---|
| W/P1_SG15-A | W/P1 | 5/5 | 0.799652 / 0.799652 | +0.000000 | W/P1 |
| W/P1_SG15-B | W/P1 | 5/5 | 0.799652 / 0.799652 | +0.000000 | W/P1 |
| W/SG11/trim20/rho0p80_SG15-A | W/SG11/trim20/rho0p80 | 5/5 | 0.798400 / 0.798400 | +0.000000 | W/SG11/trim20/rho0p80 |
| W/SG11/trim20/rho0p80_SG15-B | W/SG11/trim20/rho0p80 | 5/5 | 0.798400 / 0.798400 | +0.000000 | W/SG11/trim20/rho0p80 |
| W/SG12/top50_SG15-A | W/SG12/top50 | 5/5 | 0.796658 / 0.796658 | +0.000000 | W/SG12/top50 |
| W/SG12/top50_SG15-B | W/SG12/top50 | 5/5 | 0.796658 / 0.796658 | +0.000000 | W/SG12/top50 |

Forced candidate OOF and protocol-selected OOF are distinct. All five fold winners, thresholds, class IoUs, deltas, paired 20k-bootstrap intervals and improved/worsened/tied counts are in summary.json and fold_results.json.

## Mechanism diagnostics

```json
{
  "voc": {
    "oracle_mean_image_iou": 0.8858825673928974,
    "legacy_SG13_nearest64_GT_oracle_mean_image_iou": 0.8751898845476893,
    "oracle_note": "Main diagnostic GT fractions are evaluated over annotation-resolution regions, as are prior occupancies. Legacy nearest64-GT SG13 oracle is also retained, never used by prediction.",
    "region_selected_definition": "at least 50% output occupancy in nonempty region; for majority projection equals its chosen label, for B a diagnostic summary of mixed regions",
    "cells": {
      "W/P1/SG15-A025": {
        "region_quality": {
          "foreground_regions_per_image": 18.979591836734695,
          "region_precision": 0.8311827956989247,
          "region_recall": 0.9549104385423101,
          "region_f1": 0.8887611382581201,
          "oracle_positive_regions_missed": 365,
          "oracle_negative_regions_selected": 1570,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.751020408163264,
          "region_precision": 0.8721545182800644,
          "region_recall": 0.9371216800494132,
          "region_f1": 0.9034716846305008,
          "oracle_positive_regions_missed": 509,
          "oracle_negative_regions_selected": 1112,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8032513708155546,
        "refined_mean_image_iou": 0.7704781839905209,
        "oracle_gap_before": 0.08263119657734275,
        "oracle_gap_after": 0.11540438340237646,
        "fraction_oracle_gap_closed": -0.396620019829412,
        "changes": {
          "all": {
            "FN_to_TP": 1144369,
            "FP_to_TN": 461666,
            "TP_to_FN": 314484,
            "TN_to_FP": 3206163,
            "useful": 1606035,
            "harmful": 3520647
          },
          "high_FG": {
            "FN_to_TP": 251913,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 593863,
            "useful": 251913,
            "harmful": 593863
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 461666,
            "TP_to_FN": 314484,
            "TN_to_FP": 0,
            "useful": 461666,
            "harmful": 314484
          },
          "mixed": {
            "FN_to_TP": 892456,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 2612300,
            "useful": 892456,
            "harmful": 2612300
          }
        }
      },
      "W/P1/SG15-A050": {
        "region_quality": {
          "foreground_regions_per_image": 17.751020408163264,
          "region_precision": 0.8721545182800644,
          "region_recall": 0.9371216800494132,
          "region_f1": 0.9034716846305008,
          "oracle_positive_regions_missed": 509,
          "oracle_negative_regions_selected": 1112,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.751020408163264,
          "region_precision": 0.8721545182800644,
          "region_recall": 0.9371216800494132,
          "region_f1": 0.9034716846305008,
          "oracle_positive_regions_missed": 509,
          "oracle_negative_regions_selected": 1112,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8032513708155546,
        "refined_mean_image_iou": 0.7880960906255791,
        "oracle_gap_before": 0.08263119657734275,
        "oracle_gap_after": 0.09778647676731833,
        "fraction_oracle_gap_closed": -0.18340869813969415,
        "changes": {
          "all": {
            "FN_to_TP": 647267,
            "FP_to_TN": 1070588,
            "TP_to_FN": 945646,
            "TN_to_FP": 1597941,
            "useful": 1717855,
            "harmful": 2543587
          },
          "high_FG": {
            "FN_to_TP": 251913,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 593863,
            "useful": 251913,
            "harmful": 593863
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 461666,
            "TP_to_FN": 314484,
            "TN_to_FP": 0,
            "useful": 461666,
            "harmful": 314484
          },
          "mixed": {
            "FN_to_TP": 395354,
            "FP_to_TN": 608922,
            "TP_to_FN": 631162,
            "TN_to_FP": 1004078,
            "useful": 1004276,
            "harmful": 1635240
          }
        }
      },
      "W/P1/SG15-A075": {
        "region_quality": {
          "foreground_regions_per_image": 16.336734693877553,
          "region_precision": 0.9084322298563398,
          "region_recall": 0.8983323038912909,
          "region_f1": 0.9033540372670807,
          "oracle_positive_regions_missed": 823,
          "oracle_negative_regions_selected": 733,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.751020408163264,
          "region_precision": 0.8721545182800644,
          "region_recall": 0.9371216800494132,
          "region_f1": 0.9034716846305008,
          "oracle_positive_regions_missed": 509,
          "oracle_negative_regions_selected": 1112,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8032513708155546,
        "refined_mean_image_iou": 0.7854716886351876,
        "oracle_gap_before": 0.08263119657734275,
        "oracle_gap_after": 0.10041087875770982,
        "fraction_oracle_gap_closed": -0.21516912397275165,
        "changes": {
          "all": {
            "FN_to_TP": 251913,
            "FP_to_TN": 2005760,
            "TP_to_FN": 2414370,
            "TN_to_FP": 593863,
            "useful": 2257673,
            "harmful": 3008233
          },
          "high_FG": {
            "FN_to_TP": 251913,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 593863,
            "useful": 251913,
            "harmful": 593863
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 461666,
            "TP_to_FN": 314484,
            "TN_to_FP": 0,
            "useful": 461666,
            "harmful": 314484
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 1544094,
            "TP_to_FN": 2099886,
            "TN_to_FP": 0,
            "useful": 1544094,
            "harmful": 2099886
          }
        }
      },
      "W/P1/SG15-B": {
        "region_quality": {
          "foreground_regions_per_image": 17.751020408163264,
          "region_precision": 0.8721545182800644,
          "region_recall": 0.9371216800494132,
          "region_f1": 0.9034716846305008,
          "oracle_positive_regions_missed": 509,
          "oracle_negative_regions_selected": 1112,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.751020408163264,
          "region_precision": 0.8721545182800644,
          "region_recall": 0.9371216800494132,
          "region_f1": 0.9034716846305008,
          "oracle_positive_regions_missed": 509,
          "oracle_negative_regions_selected": 1112,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8032513708155546,
        "refined_mean_image_iou": 0.7999224782776669,
        "oracle_gap_before": 0.08263119657734275,
        "oracle_gap_after": 0.08596008911523045,
        "fraction_oracle_gap_closed": -0.040286147070033834,
        "changes": {
          "all": {
            "FN_to_TP": 251913,
            "FP_to_TN": 461666,
            "TP_to_FN": 314484,
            "TN_to_FP": 593863,
            "useful": 713579,
            "harmful": 908347
          },
          "high_FG": {
            "FN_to_TP": 251913,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 593863,
            "useful": 251913,
            "harmful": 593863
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 461666,
            "TP_to_FN": 314484,
            "TN_to_FP": 0,
            "useful": 461666,
            "harmful": 314484
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 0,
            "useful": 0,
            "harmful": 0
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-A025": {
        "region_quality": {
          "foreground_regions_per_image": 19.055102040816326,
          "region_precision": 0.8296026560993895,
          "region_recall": 0.956886967263743,
          "region_f1": 0.8887104176227627,
          "oracle_positive_regions_missed": 349,
          "oracle_negative_regions_selected": 1591,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.675510204081633,
          "region_precision": 0.8725320401801178,
          "region_recall": 0.9335392217418159,
          "region_f1": 0.9020052518500836,
          "oracle_positive_regions_missed": 538,
          "oracle_negative_regions_selected": 1104,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.7974127767071819,
        "refined_mean_image_iou": 0.7703387745120729,
        "oracle_gap_before": 0.08846979068571548,
        "oracle_gap_after": 0.1155437928808245,
        "fraction_oracle_gap_closed": -0.30602538996941986,
        "changes": {
          "all": {
            "FN_to_TP": 1512331,
            "FP_to_TN": 534280,
            "TP_to_FN": 330144,
            "TN_to_FP": 3480200,
            "useful": 2046611,
            "harmful": 3810344
          },
          "high_FG": {
            "FN_to_TP": 379068,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 582409,
            "useful": 379068,
            "harmful": 582409
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 534280,
            "TP_to_FN": 330144,
            "TN_to_FP": 0,
            "useful": 534280,
            "harmful": 330144
          },
          "mixed": {
            "FN_to_TP": 1133263,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 2897791,
            "useful": 1133263,
            "harmful": 2897791
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-A050": {
        "region_quality": {
          "foreground_regions_per_image": 17.675510204081633,
          "region_precision": 0.8725320401801178,
          "region_recall": 0.9335392217418159,
          "region_f1": 0.9020052518500836,
          "oracle_positive_regions_missed": 538,
          "oracle_negative_regions_selected": 1104,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.675510204081633,
          "region_precision": 0.8725320401801178,
          "region_recall": 0.9335392217418159,
          "region_f1": 0.9020052518500836,
          "oracle_positive_regions_missed": 538,
          "oracle_negative_regions_selected": 1104,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.7974127767071819,
        "refined_mean_image_iou": 0.7863106949155261,
        "oracle_gap_before": 0.08846979068571548,
        "oracle_gap_after": 0.09957187247737131,
        "fraction_oracle_gap_closed": -0.1254900876966628,
        "changes": {
          "all": {
            "FN_to_TP": 822700,
            "FP_to_TN": 1281364,
            "TP_to_FN": 1067096,
            "TN_to_FP": 1675179,
            "useful": 2104064,
            "harmful": 2742275
          },
          "high_FG": {
            "FN_to_TP": 379068,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 582409,
            "useful": 379068,
            "harmful": 582409
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 534280,
            "TP_to_FN": 330144,
            "TN_to_FP": 0,
            "useful": 534280,
            "harmful": 330144
          },
          "mixed": {
            "FN_to_TP": 443632,
            "FP_to_TN": 747084,
            "TP_to_FN": 736952,
            "TN_to_FP": 1092770,
            "useful": 1190716,
            "harmful": 1829722
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-A075": {
        "region_quality": {
          "foreground_regions_per_image": 16.13877551020408,
          "region_precision": 0.9124936772888215,
          "region_recall": 0.8914144533662754,
          "region_f1": 0.9018309067049929,
          "oracle_positive_regions_missed": 879,
          "oracle_negative_regions_selected": 692,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.675510204081633,
          "region_precision": 0.8725320401801178,
          "region_recall": 0.9335392217418159,
          "region_f1": 0.9020052518500836,
          "oracle_positive_regions_missed": 538,
          "oracle_negative_regions_selected": 1104,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.7974127767071819,
        "refined_mean_image_iou": 0.7836385790866278,
        "oracle_gap_before": 0.08846979068571548,
        "oracle_gap_after": 0.10224398830626957,
        "fraction_oracle_gap_closed": -0.15569379687453141,
        "changes": {
          "all": {
            "FN_to_TP": 379068,
            "FP_to_TN": 2395582,
            "TP_to_FN": 2611356,
            "TN_to_FP": 582409,
            "useful": 2774650,
            "harmful": 3193765
          },
          "high_FG": {
            "FN_to_TP": 379068,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 582409,
            "useful": 379068,
            "harmful": 582409
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 534280,
            "TP_to_FN": 330144,
            "TN_to_FP": 0,
            "useful": 534280,
            "harmful": 330144
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 1861302,
            "TP_to_FN": 2281212,
            "TN_to_FP": 0,
            "useful": 1861302,
            "harmful": 2281212
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-B": {
        "region_quality": {
          "foreground_regions_per_image": 17.675510204081633,
          "region_precision": 0.8725320401801178,
          "region_recall": 0.9335392217418159,
          "region_f1": 0.9020052518500836,
          "oracle_positive_regions_missed": 538,
          "oracle_negative_regions_selected": 1104,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.675510204081633,
          "region_precision": 0.8725320401801178,
          "region_recall": 0.9335392217418159,
          "region_f1": 0.9020052518500836,
          "oracle_positive_regions_missed": 538,
          "oracle_negative_regions_selected": 1104,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.7974127767071819,
        "refined_mean_image_iou": 0.7977207391849875,
        "oracle_gap_before": 0.08846979068571548,
        "oracle_gap_after": 0.08816182820790985,
        "fraction_oracle_gap_closed": 0.003480990238799645,
        "changes": {
          "all": {
            "FN_to_TP": 379068,
            "FP_to_TN": 534280,
            "TP_to_FN": 330144,
            "TN_to_FP": 582409,
            "useful": 913348,
            "harmful": 912553
          },
          "high_FG": {
            "FN_to_TP": 379068,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 582409,
            "useful": 379068,
            "harmful": 582409
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 534280,
            "TP_to_FN": 330144,
            "TN_to_FP": 0,
            "useful": 534280,
            "harmful": 330144
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 0,
            "useful": 0,
            "harmful": 0
          }
        }
      },
      "W/SG12/top50/SG15-A025": {
        "region_quality": {
          "foreground_regions_per_image": 18.959183673469386,
          "region_precision": 0.8313240043057051,
          "region_recall": 0.9540457072266831,
          "region_f1": 0.8884670693126259,
          "oracle_positive_regions_missed": 372,
          "oracle_negative_regions_selected": 1567,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.744897959183675,
          "region_precision": 0.8700402530189765,
          "region_recall": 0.9345274861025324,
          "region_f1": 0.901131625967838,
          "oracle_positive_regions_missed": 530,
          "oracle_negative_regions_selected": 1130,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8005040675344561,
        "refined_mean_image_iou": 0.771210115119426,
        "oracle_gap_before": 0.08537849985844126,
        "oracle_gap_after": 0.11467245227347134,
        "fraction_oracle_gap_closed": -0.34310690002283784,
        "changes": {
          "all": {
            "FN_to_TP": 1349321,
            "FP_to_TN": 537205,
            "TP_to_FN": 310205,
            "TN_to_FP": 3215017,
            "useful": 1886526,
            "harmful": 3525222
          },
          "high_FG": {
            "FN_to_TP": 286570,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 606790,
            "useful": 286570,
            "harmful": 606790
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 537205,
            "TP_to_FN": 310205,
            "TN_to_FP": 4080,
            "useful": 537205,
            "harmful": 314285
          },
          "mixed": {
            "FN_to_TP": 1062751,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 2604147,
            "useful": 1062751,
            "harmful": 2604147
          }
        }
      },
      "W/SG12/top50/SG15-A050": {
        "region_quality": {
          "foreground_regions_per_image": 17.744897959183675,
          "region_precision": 0.8700402530189765,
          "region_recall": 0.9345274861025324,
          "region_f1": 0.901131625967838,
          "oracle_positive_regions_missed": 530,
          "oracle_negative_regions_selected": 1130,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.744897959183675,
          "region_precision": 0.8700402530189765,
          "region_recall": 0.9345274861025324,
          "region_f1": 0.901131625967838,
          "oracle_positive_regions_missed": 530,
          "oracle_negative_regions_selected": 1130,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8005040675344561,
        "refined_mean_image_iou": 0.7869587018709261,
        "oracle_gap_before": 0.08537849985844126,
        "oracle_gap_after": 0.0989238655219713,
        "fraction_oracle_gap_closed": -0.15865078077019912,
        "changes": {
          "all": {
            "FN_to_TP": 749816,
            "FP_to_TN": 1126662,
            "TP_to_FN": 981458,
            "TN_to_FP": 1637792,
            "useful": 1876478,
            "harmful": 2619250
          },
          "high_FG": {
            "FN_to_TP": 286570,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 606790,
            "useful": 286570,
            "harmful": 606790
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 538482,
            "TP_to_FN": 310288,
            "TN_to_FP": 0,
            "useful": 538482,
            "harmful": 310288
          },
          "mixed": {
            "FN_to_TP": 463246,
            "FP_to_TN": 588180,
            "TP_to_FN": 671170,
            "TN_to_FP": 1031002,
            "useful": 1051426,
            "harmful": 1702172
          }
        }
      },
      "W/SG12/top50/SG15-A075": {
        "region_quality": {
          "foreground_regions_per_image": 16.261224489795918,
          "region_precision": 0.9097640562248996,
          "region_recall": 0.8954910438542311,
          "region_f1": 0.9025711261906245,
          "oracle_positive_regions_missed": 846,
          "oracle_negative_regions_selected": 719,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.744897959183675,
          "region_precision": 0.8700402530189765,
          "region_recall": 0.9345274861025324,
          "region_f1": 0.901131625967838,
          "oracle_positive_regions_missed": 530,
          "oracle_negative_regions_selected": 1130,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8005040675344561,
        "refined_mean_image_iou": 0.784793672557623,
        "oracle_gap_before": 0.08537849985844126,
        "oracle_gap_after": 0.10108889483527439,
        "fraction_oracle_gap_closed": -0.1840087961592343,
        "changes": {
          "all": {
            "FN_to_TP": 286570,
            "FP_to_TN": 2142343,
            "TP_to_FN": 2556073,
            "TN_to_FP": 606790,
            "useful": 2428913,
            "harmful": 3162863
          },
          "high_FG": {
            "FN_to_TP": 286570,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 606790,
            "useful": 286570,
            "harmful": 606790
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 538482,
            "TP_to_FN": 310288,
            "TN_to_FP": 0,
            "useful": 538482,
            "harmful": 310288
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 1603861,
            "TP_to_FN": 2245785,
            "TN_to_FP": 0,
            "useful": 1603861,
            "harmful": 2245785
          }
        }
      },
      "W/SG12/top50/SG15-B": {
        "region_quality": {
          "foreground_regions_per_image": 17.744897959183675,
          "region_precision": 0.8700402530189765,
          "region_recall": 0.9345274861025324,
          "region_f1": 0.901131625967838,
          "oracle_positive_regions_missed": 530,
          "oracle_negative_regions_selected": 1130,
          "oracle_positive_regions": 8095
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 17.744897959183675,
          "region_precision": 0.8700402530189765,
          "region_recall": 0.9345274861025324,
          "region_f1": 0.901131625967838,
          "oracle_positive_regions_missed": 530,
          "oracle_negative_regions_selected": 1130,
          "oracle_positive_regions": 8095
        },
        "unrefined_mean_image_iou": 0.8005040675344561,
        "refined_mean_image_iou": 0.798835914290351,
        "oracle_gap_before": 0.08537849985844126,
        "oracle_gap_after": 0.08704665310254645,
        "fraction_oracle_gap_closed": -0.019538329285136272,
        "changes": {
          "all": {
            "FN_to_TP": 286570,
            "FP_to_TN": 538482,
            "TP_to_FN": 310288,
            "TN_to_FP": 606790,
            "useful": 825052,
            "harmful": 917078
          },
          "high_FG": {
            "FN_to_TP": 286570,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 606790,
            "useful": 286570,
            "harmful": 606790
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 538482,
            "TP_to_FN": 310288,
            "TN_to_FP": 0,
            "useful": 538482,
            "harmful": 310288
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 0,
            "useful": 0,
            "harmful": 0
          }
        }
      }
    }
  },
  "coco": {
    "oracle_mean_image_iou": 0.805229982393218,
    "legacy_SG13_nearest64_GT_oracle_mean_image_iou": 0.7946008680974529,
    "oracle_note": "Main diagnostic GT fractions are evaluated over annotation-resolution regions, as are prior occupancies. Legacy nearest64-GT SG13 oracle is also retained, never used by prediction.",
    "region_selected_definition": "at least 50% output occupancy in nonempty region; for majority projection equals its chosen label, for B a diagnostic summary of mixed regions",
    "cells": {
      "W/P1/SG15-A025": {
        "region_quality": {
          "foreground_regions_per_image": 14.885431400282885,
          "region_precision": 0.6313188901558343,
          "region_recall": 0.9397454031117397,
          "region_f1": 0.755257474138911,
          "oracle_positive_regions_missed": 426,
          "oracle_negative_regions_selected": 3880,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.582743988684582,
          "region_precision": 0.6772883473914402,
          "region_recall": 0.9199434229137199,
          "region_f1": 0.7801835302585018,
          "oracle_positive_regions_missed": 566,
          "oracle_negative_regions_selected": 3099,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6510521253313133,
        "refined_mean_image_iou": 0.6049967878747269,
        "oracle_gap_before": 0.15417785706190468,
        "oracle_gap_after": 0.2002331945184911,
        "fraction_oracle_gap_closed": -0.29871564136537787,
        "changes": {
          "all": {
            "FN_to_TP": 1113108,
            "FP_to_TN": 955833,
            "TP_to_FN": 326916,
            "TN_to_FP": 5576127,
            "useful": 2068941,
            "harmful": 5903043
          },
          "high_FG": {
            "FN_to_TP": 221748,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 925494,
            "useful": 221748,
            "harmful": 925494
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 955833,
            "TP_to_FN": 326916,
            "TN_to_FP": 5568,
            "useful": 955833,
            "harmful": 332484
          },
          "mixed": {
            "FN_to_TP": 891360,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 4645065,
            "useful": 891360,
            "harmful": 4645065
          }
        }
      },
      "W/P1/SG15-A050": {
        "region_quality": {
          "foreground_regions_per_image": 13.582743988684582,
          "region_precision": 0.6772883473914402,
          "region_recall": 0.9199434229137199,
          "region_f1": 0.7801835302585018,
          "oracle_positive_regions_missed": 566,
          "oracle_negative_regions_selected": 3099,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.582743988684582,
          "region_precision": 0.6772883473914402,
          "region_recall": 0.9199434229137199,
          "region_f1": 0.7801835302585018,
          "oracle_positive_regions_missed": 566,
          "oracle_negative_regions_selected": 3099,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6510521253313133,
        "refined_mean_image_iou": 0.6300131624982073,
        "oracle_gap_before": 0.15417785706190468,
        "oracle_gap_after": 0.1752168198950107,
        "fraction_oracle_gap_closed": -0.1364590430431172,
        "changes": {
          "all": {
            "FN_to_TP": 550040,
            "FP_to_TN": 2257263,
            "TP_to_FN": 1073358,
            "TN_to_FP": 2710459,
            "useful": 2807303,
            "harmful": 3783817
          },
          "high_FG": {
            "FN_to_TP": 221748,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 925494,
            "useful": 221748,
            "harmful": 925494
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 957689,
            "TP_to_FN": 326916,
            "TN_to_FP": 0,
            "useful": 957689,
            "harmful": 326916
          },
          "mixed": {
            "FN_to_TP": 328292,
            "FP_to_TN": 1299574,
            "TP_to_FN": 746442,
            "TN_to_FP": 1784965,
            "useful": 1627866,
            "harmful": 2531407
          }
        }
      },
      "W/P1/SG15-A075": {
        "region_quality": {
          "foreground_regions_per_image": 12.091937765205092,
          "region_precision": 0.7248801029360159,
          "region_recall": 0.8765205091937766,
          "region_f1": 0.79352071195339,
          "oracle_positive_regions_missed": 873,
          "oracle_negative_regions_selected": 2352,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.582743988684582,
          "region_precision": 0.6772883473914402,
          "region_recall": 0.9199434229137199,
          "region_f1": 0.7801835302585018,
          "oracle_positive_regions_missed": 566,
          "oracle_negative_regions_selected": 3099,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6510521253313133,
        "refined_mean_image_iou": 0.635342153487434,
        "oracle_gap_before": 0.15417785706190468,
        "oracle_gap_after": 0.16988782890578402,
        "fraction_oracle_gap_closed": -0.10189512387353755,
        "changes": {
          "all": {
            "FN_to_TP": 221748,
            "FP_to_TN": 4267438,
            "TP_to_FN": 2647910,
            "TN_to_FP": 925494,
            "useful": 4489186,
            "harmful": 3573404
          },
          "high_FG": {
            "FN_to_TP": 221748,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 925494,
            "useful": 221748,
            "harmful": 925494
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 957689,
            "TP_to_FN": 326916,
            "TN_to_FP": 0,
            "useful": 957689,
            "harmful": 326916
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 3309749,
            "TP_to_FN": 2320994,
            "TN_to_FP": 0,
            "useful": 3309749,
            "harmful": 2320994
          }
        }
      },
      "W/P1/SG15-B": {
        "region_quality": {
          "foreground_regions_per_image": 13.582743988684582,
          "region_precision": 0.6772883473914402,
          "region_recall": 0.9199434229137199,
          "region_f1": 0.7801835302585018,
          "oracle_positive_regions_missed": 566,
          "oracle_negative_regions_selected": 3099,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.582743988684582,
          "region_precision": 0.6772883473914402,
          "region_recall": 0.9199434229137199,
          "region_f1": 0.7801835302585018,
          "oracle_positive_regions_missed": 566,
          "oracle_negative_regions_selected": 3099,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6510521253313133,
        "refined_mean_image_iou": 0.6477144885820664,
        "oracle_gap_before": 0.15417785706190468,
        "oracle_gap_after": 0.1575154938111516,
        "fraction_oracle_gap_closed": -0.02164796432412995,
        "changes": {
          "all": {
            "FN_to_TP": 221748,
            "FP_to_TN": 957689,
            "TP_to_FN": 326916,
            "TN_to_FP": 925494,
            "useful": 1179437,
            "harmful": 1252410
          },
          "high_FG": {
            "FN_to_TP": 221748,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 925494,
            "useful": 221748,
            "harmful": 925494
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 957689,
            "TP_to_FN": 326916,
            "TN_to_FP": 0,
            "useful": 957689,
            "harmful": 326916
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 0,
            "useful": 0,
            "harmful": 0
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-A025": {
        "region_quality": {
          "foreground_regions_per_image": 14.478076379066477,
          "region_precision": 0.6414615084017194,
          "region_recall": 0.9287128712871288,
          "region_f1": 0.7588119727262223,
          "oracle_positive_regions_missed": 504,
          "oracle_negative_regions_selected": 3670,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.033946251768034,
          "region_precision": 0.6906131307650569,
          "region_recall": 0.9001414427157002,
          "region_f1": 0.7815781393920785,
          "oracle_positive_regions_missed": 706,
          "oracle_negative_regions_selected": 2851,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6521182759251232,
        "refined_mean_image_iou": 0.6100519534912684,
        "oracle_gap_before": 0.1531117064680948,
        "oracle_gap_after": 0.19517802890194957,
        "fraction_oracle_gap_closed": -0.27474269214431685,
        "changes": {
          "all": {
            "FN_to_TP": 1507135,
            "FP_to_TN": 1043795,
            "TP_to_FN": 378723,
            "TN_to_FP": 5826791,
            "useful": 2550930,
            "harmful": 6205514
          },
          "high_FG": {
            "FN_to_TP": 334146,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 966912,
            "useful": 334146,
            "harmful": 966912
          },
          "high_BG": {
            "FN_to_TP": 4942,
            "FP_to_TN": 1043795,
            "TP_to_FN": 378723,
            "TN_to_FP": 1634,
            "useful": 1048737,
            "harmful": 380357
          },
          "mixed": {
            "FN_to_TP": 1168047,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 4858245,
            "useful": 1168047,
            "harmful": 4858245
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-A050": {
        "region_quality": {
          "foreground_regions_per_image": 13.033946251768034,
          "region_precision": 0.6906131307650569,
          "region_recall": 0.9001414427157002,
          "region_f1": 0.7815781393920785,
          "oracle_positive_regions_missed": 706,
          "oracle_negative_regions_selected": 2851,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.033946251768034,
          "region_precision": 0.6906131307650569,
          "region_recall": 0.9001414427157002,
          "region_f1": 0.7815781393920785,
          "oracle_positive_regions_missed": 706,
          "oracle_negative_regions_selected": 2851,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6521182759251232,
        "refined_mean_image_iou": 0.6329536610472555,
        "oracle_gap_before": 0.1531117064680948,
        "oracle_gap_after": 0.1722763213459625,
        "fraction_oracle_gap_closed": -0.12516753499747055,
        "changes": {
          "all": {
            "FN_to_TP": 787369,
            "FP_to_TN": 2423526,
            "TP_to_FN": 1247126,
            "TN_to_FP": 2737155,
            "useful": 3210895,
            "harmful": 3984281
          },
          "high_FG": {
            "FN_to_TP": 334146,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 966912,
            "useful": 334146,
            "harmful": 966912
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 1044049,
            "TP_to_FN": 380661,
            "TN_to_FP": 0,
            "useful": 1044049,
            "harmful": 380661
          },
          "mixed": {
            "FN_to_TP": 453223,
            "FP_to_TN": 1379477,
            "TP_to_FN": 866465,
            "TN_to_FP": 1770243,
            "useful": 1832700,
            "harmful": 2636708
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-A075": {
        "region_quality": {
          "foreground_regions_per_image": 11.516265912305517,
          "region_precision": 0.7397445345124048,
          "region_recall": 0.8519094766619519,
          "region_f1": 0.7918748356560611,
          "oracle_positive_regions_missed": 1047,
          "oracle_negative_regions_selected": 2119,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.033946251768034,
          "region_precision": 0.6906131307650569,
          "region_recall": 0.9001414427157002,
          "region_f1": 0.7815781393920785,
          "oracle_positive_regions_missed": 706,
          "oracle_negative_regions_selected": 2851,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6521182759251232,
        "refined_mean_image_iou": 0.634529737070215,
        "oracle_gap_before": 0.1531117064680948,
        "oracle_gap_after": 0.17070024532300299,
        "fraction_oracle_gap_closed": -0.11487389998212358,
        "changes": {
          "all": {
            "FN_to_TP": 334146,
            "FP_to_TN": 4509727,
            "TP_to_FN": 2943603,
            "TN_to_FP": 966912,
            "useful": 4843873,
            "harmful": 3910515
          },
          "high_FG": {
            "FN_to_TP": 334146,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 966912,
            "useful": 334146,
            "harmful": 966912
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 1044049,
            "TP_to_FN": 380661,
            "TN_to_FP": 0,
            "useful": 1044049,
            "harmful": 380661
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 3465678,
            "TP_to_FN": 2562942,
            "TN_to_FP": 0,
            "useful": 3465678,
            "harmful": 2562942
          }
        }
      },
      "W/SG11/trim20/rho0p80/SG15-B": {
        "region_quality": {
          "foreground_regions_per_image": 13.033946251768034,
          "region_precision": 0.6906131307650569,
          "region_recall": 0.9001414427157002,
          "region_f1": 0.7815781393920785,
          "oracle_positive_regions_missed": 706,
          "oracle_negative_regions_selected": 2851,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.033946251768034,
          "region_precision": 0.6906131307650569,
          "region_recall": 0.9001414427157002,
          "region_f1": 0.7815781393920785,
          "oracle_positive_regions_missed": 706,
          "oracle_negative_regions_selected": 2851,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6521182759251232,
        "refined_mean_image_iou": 0.6512791449053763,
        "oracle_gap_before": 0.1531117064680948,
        "oracle_gap_after": 0.15395083748784166,
        "fraction_oracle_gap_closed": -0.0054805151030154625,
        "changes": {
          "all": {
            "FN_to_TP": 334146,
            "FP_to_TN": 1044049,
            "TP_to_FN": 380661,
            "TN_to_FP": 966912,
            "useful": 1378195,
            "harmful": 1347573
          },
          "high_FG": {
            "FN_to_TP": 334146,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 966912,
            "useful": 334146,
            "harmful": 966912
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 1044049,
            "TP_to_FN": 380661,
            "TN_to_FP": 0,
            "useful": 1044049,
            "harmful": 380661
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 0,
            "useful": 0,
            "harmful": 0
          }
        }
      },
      "W/SG12/top50/SG15-A025": {
        "region_quality": {
          "foreground_regions_per_image": 14.782178217821782,
          "region_precision": 0.6351545306669218,
          "region_recall": 0.9388967468175389,
          "region_f1": 0.7577193082586612,
          "oracle_positive_regions_missed": 432,
          "oracle_negative_regions_selected": 3813,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.346534653465346,
          "region_precision": 0.6867316659601526,
          "region_recall": 0.9165487977369166,
          "region_f1": 0.7851690294438386,
          "oracle_positive_regions_missed": 590,
          "oracle_negative_regions_selected": 2956,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6514281579666902,
        "refined_mean_image_iou": 0.605314291741782,
        "oracle_gap_before": 0.1538018244265278,
        "oracle_gap_after": 0.199915690651436,
        "fraction_oracle_gap_closed": -0.29982652284425354,
        "changes": {
          "all": {
            "FN_to_TP": 1263859,
            "FP_to_TN": 1016077,
            "TP_to_FN": 339874,
            "TN_to_FP": 5941969,
            "useful": 2279936,
            "harmful": 6281843
          },
          "high_FG": {
            "FN_to_TP": 247809,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 929788,
            "useful": 247809,
            "harmful": 929788
          },
          "high_BG": {
            "FN_to_TP": 1143,
            "FP_to_TN": 1016077,
            "TP_to_FN": 339874,
            "TN_to_FP": 5289,
            "useful": 1017220,
            "harmful": 345163
          },
          "mixed": {
            "FN_to_TP": 1014907,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 5006892,
            "useful": 1014907,
            "harmful": 5006892
          }
        }
      },
      "W/SG12/top50/SG15-A050": {
        "region_quality": {
          "foreground_regions_per_image": 13.346534653465346,
          "region_precision": 0.6867316659601526,
          "region_recall": 0.9165487977369166,
          "region_f1": 0.7851690294438386,
          "oracle_positive_regions_missed": 590,
          "oracle_negative_regions_selected": 2956,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.346534653465346,
          "region_precision": 0.6867316659601526,
          "region_recall": 0.9165487977369166,
          "region_f1": 0.7851690294438386,
          "oracle_positive_regions_missed": 590,
          "oracle_negative_regions_selected": 2956,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6514281579666902,
        "refined_mean_image_iou": 0.6325272435170017,
        "oracle_gap_before": 0.1538018244265278,
        "oracle_gap_after": 0.17270273887621623,
        "fraction_oracle_gap_closed": -0.12289135398857082,
        "changes": {
          "all": {
            "FN_to_TP": 651942,
            "FP_to_TN": 2505901,
            "TP_to_FN": 1140368,
            "TN_to_FP": 2728044,
            "useful": 3157843,
            "harmful": 3868412
          },
          "high_FG": {
            "FN_to_TP": 247809,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 929788,
            "useful": 247809,
            "harmful": 929788
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 1016516,
            "TP_to_FN": 341579,
            "TN_to_FP": 0,
            "useful": 1016516,
            "harmful": 341579
          },
          "mixed": {
            "FN_to_TP": 404133,
            "FP_to_TN": 1489385,
            "TP_to_FN": 798789,
            "TN_to_FP": 1798256,
            "useful": 1893518,
            "harmful": 2597045
          }
        }
      },
      "W/SG12/top50/SG15-A075": {
        "region_quality": {
          "foreground_regions_per_image": 11.847241867043847,
          "region_precision": 0.7349570200573066,
          "region_recall": 0.8707213578500708,
          "region_f1": 0.7970995727049074,
          "oracle_positive_regions_missed": 914,
          "oracle_negative_regions_selected": 2220,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.346534653465346,
          "region_precision": 0.6867316659601526,
          "region_recall": 0.9165487977369166,
          "region_f1": 0.7851690294438386,
          "oracle_positive_regions_missed": 590,
          "oracle_negative_regions_selected": 2956,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6514281579666902,
        "refined_mean_image_iou": 0.6362153078436461,
        "oracle_gap_before": 0.1538018244265278,
        "oracle_gap_after": 0.16901467454957186,
        "fraction_oracle_gap_closed": -0.09891202643250412,
        "changes": {
          "all": {
            "FN_to_TP": 247809,
            "FP_to_TN": 4530175,
            "TP_to_FN": 2824105,
            "TN_to_FP": 929788,
            "useful": 4777984,
            "harmful": 3753893
          },
          "high_FG": {
            "FN_to_TP": 247809,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 929788,
            "useful": 247809,
            "harmful": 929788
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 1016516,
            "TP_to_FN": 341579,
            "TN_to_FP": 0,
            "useful": 1016516,
            "harmful": 341579
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 3513659,
            "TP_to_FN": 2482526,
            "TN_to_FP": 0,
            "useful": 3513659,
            "harmful": 2482526
          }
        }
      },
      "W/SG12/top50/SG15-B": {
        "region_quality": {
          "foreground_regions_per_image": 13.346534653465346,
          "region_precision": 0.6867316659601526,
          "region_recall": 0.9165487977369166,
          "region_f1": 0.7851690294438386,
          "oracle_positive_regions_missed": 590,
          "oracle_negative_regions_selected": 2956,
          "oracle_positive_regions": 7070
        },
        "prior_region_quality": {
          "foreground_regions_per_image": 13.346534653465346,
          "region_precision": 0.6867316659601526,
          "region_recall": 0.9165487977369166,
          "region_f1": 0.7851690294438386,
          "oracle_positive_regions_missed": 590,
          "oracle_negative_regions_selected": 2956,
          "oracle_positive_regions": 7070
        },
        "unrefined_mean_image_iou": 0.6514281579666902,
        "refined_mean_image_iou": 0.649099841890565,
        "oracle_gap_before": 0.1538018244265278,
        "oracle_gap_after": 0.15613014050265295,
        "fraction_oracle_gap_closed": -0.015138416496726355,
        "changes": {
          "all": {
            "FN_to_TP": 247809,
            "FP_to_TN": 1016516,
            "TP_to_FN": 341579,
            "TN_to_FP": 929788,
            "useful": 1264325,
            "harmful": 1271367
          },
          "high_FG": {
            "FN_to_TP": 247809,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 929788,
            "useful": 247809,
            "harmful": 929788
          },
          "high_BG": {
            "FN_to_TP": 0,
            "FP_to_TN": 1016516,
            "TP_to_FN": 341579,
            "TN_to_FP": 0,
            "useful": 1016516,
            "harmful": 341579
          },
          "mixed": {
            "FN_to_TP": 0,
            "FP_to_TN": 0,
            "TP_to_FN": 0,
            "TN_to_FP": 0,
            "useful": 0,
            "harmful": 0
          }
        }
      }
    }
  }
}
```
