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

## Five folds and forced OOF comparisons

```json
{
  "screens": {
    "primary": {
      "reference": "W/P1+R2",
      "candidates": [
        "W/SG12/top50+R2"
      ],
      "folds": [
        {
          "fold": 0,
          "chosen": "W/P1+R2",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8124570594443471,
          "train_miou_reference": 0.8124570594443471,
          "heldout_miou_chosen": 0.8206285684612318,
          "heldout_miou_reference": 0.8206285684612318
        },
        {
          "fold": 1,
          "chosen": "W/P1+R2",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8064126586745404,
          "train_miou_reference": 0.8064126586745404,
          "heldout_miou_chosen": 0.8412213472653388,
          "heldout_miou_reference": 0.8412213472653388
        },
        {
          "fold": 2,
          "chosen": "W/P1+R2",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8138403926993096,
          "train_miou_reference": 0.8138403926993096,
          "heldout_miou_chosen": 0.8138492414093419,
          "heldout_miou_reference": 0.8138492414093419
        },
        {
          "fold": 3,
          "chosen": "W/P1+R2",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8178784564240132,
          "train_miou_reference": 0.8178784564240132,
          "heldout_miou_chosen": 0.7986398595222017,
          "heldout_miou_reference": 0.7986398595222017
        },
        {
          "fold": 4,
          "chosen": "W/P1+R2",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8158292413468764,
          "train_miou_reference": 0.8158292413468764,
          "heldout_miou_chosen": 0.7994051136990514,
          "heldout_miou_reference": 0.7994051136990514
        }
      ],
      "selection_counts": {
        "W/P1+R2": 5
      },
      "vote_winner": "W/P1+R2",
      "vote_winner_votes": 5,
      "frozen": "W/P1+R2",
      "freeze_reason": "reference won the fold vote",
      "oof_miou_selection": 0.8132087583619251,
      "oof_miou_reference": 0.8132087583619251,
      "oof_gain": 0.0,
      "full_threshold_frozen": 0.45,
      "full_threshold_reference": 0.45,
      "full_miou_frozen": 0.8132087583619251,
      "full_miou_reference": 0.8132087583619251,
      "paired_oof": {
        "mean": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "better": 0,
        "worse": 0,
        "ties": 490
      }
    }
  },
  "forced_cell_oof": {
    "W/REF": {
      "miou": 0.7872797008862745,
      "folds": [
        {
          "fold": 0,
          "threshold": 0.425,
          "heldout_miou": 0.7854325094473272
        },
        {
          "fold": 1,
          "threshold": 0.425,
          "heldout_miou": 0.8114630875761657
        },
        {
          "fold": 2,
          "threshold": 0.425,
          "heldout_miou": 0.7859289428172378
        },
        {
          "fold": 3,
          "threshold": 0.425,
          "heldout_miou": 0.7802224546432177
        },
        {
          "fold": 4,
          "threshold": 0.45,
          "heldout_miou": 0.7751596960962914
        }
      ],
      "mean_image_iou": 0.7882113942364494
    },
    "W/P1": {
      "miou": 0.7996523496064184,
      "folds": [
        {
          "fold": 0,
          "threshold": 0.45,
          "heldout_miou": 0.8043717755713968
        },
        {
          "fold": 1,
          "threshold": 0.45,
          "heldout_miou": 0.8236552709257694
        },
        {
          "fold": 2,
          "threshold": 0.45,
          "heldout_miou": 0.8023293971713426
        },
        {
          "fold": 3,
          "threshold": 0.45,
          "heldout_miou": 0.7905416055541602
        },
        {
          "fold": 4,
          "threshold": 0.475,
          "heldout_miou": 0.7818058154631513
        }
      ],
      "mean_image_iou": 0.8025622117538839
    },
    "W/SG12/top50": {
      "miou": 0.796657782422708,
      "folds": [
        {
          "fold": 0,
          "threshold": 0.45,
          "heldout_miou": 0.7980079201556065
        },
        {
          "fold": 1,
          "threshold": 0.45,
          "heldout_miou": 0.8210694391882276
        },
        {
          "fold": 2,
          "threshold": 0.45,
          "heldout_miou": 0.7957113805320305
        },
        {
          "fold": 3,
          "threshold": 0.45,
          "heldout_miou": 0.7904002140640006
        },
        {
          "fold": 4,
          "threshold": 0.475,
          "heldout_miou": 0.7811817777136335
        }
      ],
      "mean_image_iou": 0.7992673340664536
    },
    "W/P1+R2": {
      "miou": 0.8132087583619251,
      "folds": [
        {
          "fold": 0,
          "threshold": 0.45,
          "heldout_miou": 0.8206285684612318
        },
        {
          "fold": 1,
          "threshold": 0.45,
          "heldout_miou": 0.8412213472653388
        },
        {
          "fold": 2,
          "threshold": 0.45,
          "heldout_miou": 0.8138492414093419
        },
        {
          "fold": 3,
          "threshold": 0.45,
          "heldout_miou": 0.7986398595222017
        },
        {
          "fold": 4,
          "threshold": 0.45,
          "heldout_miou": 0.7994051136990514
        }
      ],
      "mean_image_iou": 0.8189078805804186
    },
    "W/SG12/top50+R2": {
      "miou": 0.8115260973784355,
      "folds": [
        {
          "fold": 0,
          "threshold": 0.425,
          "heldout_miou": 0.8104248152238318
        },
        {
          "fold": 1,
          "threshold": 0.425,
          "heldout_miou": 0.8398491268834439
        },
        {
          "fold": 2,
          "threshold": 0.425,
          "heldout_miou": 0.8083708573123968
        },
        {
          "fold": 3,
          "threshold": 0.425,
          "heldout_miou": 0.8000845826039586
        },
        {
          "fold": 4,
          "threshold": 0.45,
          "heldout_miou": 0.8044536298238001
        }
      ],
      "mean_image_iou": 0.81681450511936
    }
  },
  "forced_oof_comparisons": {
    "primary": {
      "W/SG12/top50+R2": {
        "candidate_miou": 0.8115260973784355,
        "reference_miou": 0.8132087583619251,
        "delta": -0.00168266098348957,
        "paired_mean_image_iou": {
          "mean": -0.002093375461058518,
          "ci_low": -0.005592523360350158,
          "ci_high": 0.0017953603302772938,
          "better": 178,
          "worse": 312,
          "ties": 0
        }
      }
    }
  },
  "screen_frozen_transfer": {
    "primary": {
      "reference": "W/P1+R2",
      "frozen": "W/P1+R2",
      "reference_threshold": 0.45,
      "frozen_threshold": 0.45,
      "reference_coco_miou": 0.6371407980966601,
      "frozen_coco_miou": 0.6371407980966601
    }
  }
}
```

## All class results

dataset,cell,reference,class,threshold,iou,delta
voc,W/REF,W/P1+R2,aeroplane,0.425,0.8352635722287826,-0.0609579926782724
voc,W/REF,W/P1+R2,bicycle,0.425,0.737438165800942,-0.024962087471930405
voc,W/REF,W/P1+R2,bird,0.425,0.8403247874482437,-0.04340981949147582
voc,W/REF,W/P1+R2,boat,0.425,0.8124661032715629,-0.02718817139690677
voc,W/REF,W/P1+R2,bottle,0.425,0.8561283615303112,-0.020593507602690297
voc,W/REF,W/P1+R2,bus,0.425,0.9089213701956407,0.0031402277858093575
voc,W/REF,W/P1+R2,car,0.425,0.8254276616207066,-0.0016719599785737227
voc,W/REF,W/P1+R2,cat,0.425,0.8131546400743833,-0.019946789285609223
voc,W/REF,W/P1+R2,chair,0.425,0.629114721949853,-0.14688015064266524
voc,W/REF,W/P1+R2,cow,0.425,0.8426602356685133,-0.012676202641250356
voc,W/REF,W/P1+R2,dining-table,0.425,0.45270798975063226,0.00230870776918507
voc,W/REF,W/P1+R2,dog,0.425,0.9028583590477207,-0.012124890453722004
voc,W/REF,W/P1+R2,horse,0.425,0.8655564550590958,-0.030146246416355438
voc,W/REF,W/P1+R2,motorbike,0.425,0.839833950438049,-0.008830557902045677
voc,W/REF,W/P1+R2,person,0.425,0.7057082886035627,-0.032171841909516496
voc,W/REF,W/P1+R2,potted-plant,0.425,0.7310806715606982,-0.020518895855951902
voc,W/REF,W/P1+R2,sheep,0.425,0.8001454640174295,-0.010097270225231214
voc,W/REF,W/P1+R2,sofa,0.425,0.9368351592460529,-0.021117851699374746
voc,W/REF,W/P1+R2,train,0.425,0.6433302895856994,-0.02114798221285974
voc,W/REF,W/P1+R2,tv,0.425,0.7932418189177365,0.01701618108655345
coco,W/REF,W/P1+R2,aeroplane,0.425,0.5639089672172332,-0.06997800867906012
coco,W/REF,W/P1+R2,bicycle,0.425,0.5813415797451317,0.005729524193867852
coco,W/REF,W/P1+R2,bird,0.425,0.7128468945430074,-0.04304546769218176
coco,W/REF,W/P1+R2,boat,0.425,0.5174657041725186,-0.009120730462067073
coco,W/REF,W/P1+R2,bottle,0.425,0.45657259822166135,-0.021790733190009404
coco,W/REF,W/P1+R2,bus,0.425,0.9200937035017466,0.014094357308840677
coco,W/REF,W/P1+R2,car,0.425,0.5682754030772628,0.009931867950716655
coco,W/REF,W/P1+R2,cat,0.425,0.8286793983910459,-0.03083006150571388
coco,W/REF,W/P1+R2,chair,0.425,0.4190301468853303,-0.031363935257959275
coco,W/REF,W/P1+R2,cow,0.425,0.8005908527741239,-0.041444783694107645
coco,W/REF,W/P1+R2,dining table,0.425,0.3742098228693574,0.008892366057342838
coco,W/REF,W/P1+R2,dog,0.425,0.7450744893666833,-0.011099648008357055
coco,W/REF,W/P1+R2,horse,0.425,0.6354861139057686,-0.037518531130501964
coco,W/REF,W/P1+R2,motorbike,0.425,0.6175213013121482,-0.011601368524056466
coco,W/REF,W/P1+R2,person,0.425,0.21896664030149351,-0.01241808444235687
coco,W/REF,W/P1+R2,potted plant,0.425,0.6430209450112684,-0.05348045520654332
coco,W/REF,W/P1+R2,sheep,0.425,0.8455591644890148,-0.02921567519684154
coco,W/REF,W/P1+R2,sofa,0.425,0.5901932607758212,-0.01483803407405937
coco,W/REF,W/P1+R2,train,0.425,0.6075123829233591,-0.017608657417982565
coco,W/REF,W/P1+R2,tv,0.425,0.6969191997964193,-0.0028413336817765256
voc,W/P1,W/P1+R2,aeroplane,0.45,0.8667233064113781,-0.029498258495676888
voc,W/P1,W/P1+R2,bicycle,0.45,0.7479688100389947,-0.014431443233877683
voc,W/P1,W/P1+R2,bird,0.45,0.8393990307429229,-0.04433557619679662
voc,W/P1,W/P1+R2,boat,0.45,0.8364798244241098,-0.0031744502443598055
voc,W/P1,W/P1+R2,bottle,0.45,0.8641614408977321,-0.012560428235269483
voc,W/P1,W/P1+R2,bus,0.45,0.9048342755779042,-0.000946866831927129
voc,W/P1,W/P1+R2,car,0.45,0.825555626602769,-0.0015439949965112554
voc,W/P1,W/P1+R2,cat,0.45,0.8257994172400851,-0.007302012119907442
voc,W/P1,W/P1+R2,chair,0.45,0.7090314096326927,-0.06696346295982547
voc,W/P1,W/P1+R2,cow,0.45,0.8438618425543178,-0.011474595755445849
voc,W/P1,W/P1+R2,dining-table,0.45,0.44906475758614767,-0.001334524395299519
voc,W/P1,W/P1+R2,dog,0.45,0.9021220885212011,-0.012861160980241593
voc,W/P1,W/P1+R2,horse,0.45,0.8745240110314191,-0.0211786904440322
voc,W/P1,W/P1+R2,motorbike,0.45,0.8506338393185044,0.001969330978409789
voc,W/P1,W/P1+R2,person,0.45,0.7336328856928452,-0.004247244820233975
voc,W/P1,W/P1+R2,potted-plant,0.45,0.7489042955027874,-0.0026952719138627623
voc,W/P1,W/P1+R2,sheep,0.45,0.7993203646184205,-0.010922369624240202
voc,W/P1,W/P1+R2,sofa,0.45,0.9482479174015093,-0.009705093543918353
voc,W/P1,W/P1+R2,train,0.45,0.6625303839241307,-0.0019478878744284556
voc,W/P1,W/P1+R2,tv,0.45,0.7788978034924409,0.0026721656612578304
coco,W/P1,W/P1+R2,aeroplane,0.45,0.587178187123291,-0.04670878877300233
coco,W/P1,W/P1+R2,bicycle,0.45,0.5689240406859062,-0.006688014865357639
coco,W/P1,W/P1+R2,bird,0.45,0.7223834216586578,-0.03350894057653142
coco,W/P1,W/P1+R2,boat,0.45,0.5171934129200428,-0.009393021714542815
coco,W/P1,W/P1+R2,bottle,0.45,0.46406995987944305,-0.014293371532227706
coco,W/P1,W/P1+R2,bus,0.45,0.9054070266712473,-0.000592319521658613
coco,W/P1,W/P1+R2,car,0.45,0.5581137523767445,-0.00022978274980156588
coco,W/P1,W/P1+R2,cat,0.45,0.8463892629151284,-0.01312019698163136
coco,W/P1,W/P1+R2,chair,0.45,0.43568564859445374,-0.014708433548835831
coco,W/P1,W/P1+R2,cow,0.45,0.8143433950195997,-0.027692241448631805
coco,W/P1,W/P1+R2,dining table,0.45,0.3862348871920122,0.02091743037999766
coco,W/P1,W/P1+R2,dog,0.45,0.7434133452977393,-0.012760792077301075
coco,W/P1,W/P1+R2,horse,0.45,0.6454537308929894,-0.027550914143281147
coco,W/P1,W/P1+R2,motorbike,0.45,0.6184382855637731,-0.010684384272431569
coco,W/P1,W/P1+R2,person,0.45,0.23539629463971476,0.004011569895864375
coco,W/P1,W/P1+R2,potted plant,0.45,0.6699416478955144,-0.026559752322297348
coco,W/P1,W/P1+R2,sheep,0.45,0.8627351702470436,-0.012039669438812739
coco,W/P1,W/P1+R2,sofa,0.45,0.6020553643144004,-0.0029759305354801757
coco,W/P1,W/P1+R2,train,0.45,0.614498858788715,-0.01062218155262662
coco,W/P1,W/P1+R2,tv,0.45,0.691562050139881,-0.008198483338314744
voc,W/SG12/top50,W/P1+R2,aeroplane,0.45,0.8551831673256742,-0.041038397581380726
voc,W/SG12/top50,W/P1+R2,bicycle,0.45,0.7452900929578082,-0.017110160315064182
voc,W/SG12/top50,W/P1+R2,bird,0.45,0.8564873864040521,-0.027247220535667416
voc,W/SG12/top50,W/P1+R2,boat,0.45,0.8349081845199188,-0.004746090148550874
voc,W/SG12/top50,W/P1+R2,bottle,0.45,0.8650743004251582,-0.011647568707843337
voc,W/SG12/top50,W/P1+R2,bus,0.45,0.9149153182349083,0.009134175825077029
voc,W/SG12/top50,W/P1+R2,car,0.45,0.8331636144818277,0.006063992882547442
voc,W/SG12/top50,W/P1+R2,cat,0.45,0.8088836400489084,-0.024217789311084092
voc,W/SG12/top50,W/P1+R2,chair,0.45,0.6774630278879507,-0.09853184470456755
voc,W/SG12/top50,W/P1+R2,cow,0.45,0.8456211706998886,-0.009715267609875067
voc,W/SG12/top50,W/P1+R2,dining-table,0.45,0.4615386029282244,0.01113932094677722
voc,W/SG12/top50,W/P1+R2,dog,0.45,0.9021113160917039,-0.0128719334097388
voc,W/SG12/top50,W/P1+R2,horse,0.45,0.8775036538322383,-0.018199047643213007
voc,W/SG12/top50,W/P1+R2,motorbike,0.45,0.8500637872967285,0.0013992789566338315
voc,W/SG12/top50,W/P1+R2,person,0.45,0.7225903577327976,-0.015289772780281652
voc,W/SG12/top50,W/P1+R2,potted-plant,0.45,0.7340156136998037,-0.0175839537168464
voc,W/SG12/top50,W/P1+R2,sheep,0.45,0.7844944353009128,-0.025748298941747905
voc,W/SG12/top50,W/P1+R2,sofa,0.45,0.9477164299458083,-0.010236580999619327
voc,W/SG12/top50,W/P1+R2,train,0.45,0.669341712707643,0.00486344090908386
voc,W/SG12/top50,W/P1+R2,tv,0.45,0.7848025248388913,0.008576887007708311
coco,W/SG12/top50,W/P1+R2,aeroplane,0.45,0.5900271553321144,-0.04385982056417892
coco,W/SG12/top50,W/P1+R2,bicycle,0.45,0.5803237423280837,0.004711686776819879
coco,W/SG12/top50,W/P1+R2,bird,0.45,0.7159952205631942,-0.039897141671994985
coco,W/SG12/top50,W/P1+R2,boat,0.45,0.530093329622414,0.0035068949878283018
coco,W/SG12/top50,W/P1+R2,bottle,0.45,0.4730029835566697,-0.0053603478550010575
coco,W/SG12/top50,W/P1+R2,bus,0.45,0.9178582139706918,0.011858867777785886
coco,W/SG12/top50,W/P1+R2,car,0.45,0.5629056930097877,0.004562157883241547
coco,W/SG12/top50,W/P1+R2,cat,0.45,0.8387175174664953,-0.0207919424302645
coco,W/SG12/top50,W/P1+R2,chair,0.45,0.42745203301870277,-0.022942049124586805
coco,W/SG12/top50,W/P1+R2,cow,0.45,0.8159134906457183,-0.026122145822513265
coco,W/SG12/top50,W/P1+R2,dining table,0.45,0.3807444032975091,0.01542694648549453
coco,W/SG12/top50,W/P1+R2,dog,0.45,0.7460987487672033,-0.010075388607837077
coco,W/SG12/top50,W/P1+R2,horse,0.45,0.6540135143737305,-0.0189911306625401
coco,W/SG12/top50,W/P1+R2,motorbike,0.45,0.6275318139701992,-0.0015908558660054917
coco,W/SG12/top50,W/P1+R2,person,0.45,0.23774273358530068,0.006358008841450297
coco,W/SG12/top50,W/P1+R2,potted plant,0.45,0.660547683027553,-0.03595371719025875
coco,W/SG12/top50,W/P1+R2,sheep,0.45,0.8677052584529141,-0.0070695812329422125
coco,W/SG12/top50,W/P1+R2,sofa,0.45,0.5984718293318083,-0.006559465518072294
coco,W/SG12/top50,W/P1+R2,train,0.45,0.6181185621875782,-0.007002478153763425
coco,W/SG12/top50,W/P1+R2,tv,0.45,0.7002062740475652,0.00044574056936941275
voc,W/P1+R2,W/P1+R2,aeroplane,0.45,0.896221564907055,0.0
voc,W/P1+R2,W/P1+R2,bicycle,0.45,0.7624002532728724,0.0
voc,W/P1+R2,W/P1+R2,bird,0.45,0.8837346069397195,0.0
voc,W/P1+R2,W/P1+R2,boat,0.45,0.8396542746684696,0.0
voc,W/P1+R2,W/P1+R2,bottle,0.45,0.8767218691330015,0.0
voc,W/P1+R2,W/P1+R2,bus,0.45,0.9057811424098313,0.0
voc,W/P1+R2,W/P1+R2,car,0.45,0.8270996215992803,0.0
voc,W/P1+R2,W/P1+R2,cat,0.45,0.8331014293599925,0.0
voc,W/P1+R2,W/P1+R2,chair,0.45,0.7759948725925182,0.0
voc,W/P1+R2,W/P1+R2,cow,0.45,0.8553364383097637,0.0
voc,W/P1+R2,W/P1+R2,dining-table,0.45,0.4503992819814472,0.0
voc,W/P1+R2,W/P1+R2,dog,0.45,0.9149832495014427,0.0
voc,W/P1+R2,W/P1+R2,horse,0.45,0.8957027014754513,0.0
voc,W/P1+R2,W/P1+R2,motorbike,0.45,0.8486645083400947,0.0
voc,W/P1+R2,W/P1+R2,person,0.45,0.7378801305130792,0.0
voc,W/P1+R2,W/P1+R2,potted-plant,0.45,0.7515995674166501,0.0
voc,W/P1+R2,W/P1+R2,sheep,0.45,0.8102427342426607,0.0
voc,W/P1+R2,W/P1+R2,sofa,0.45,0.9579530109454276,0.0
voc,W/P1+R2,W/P1+R2,train,0.45,0.6644782717985591,0.0
voc,W/P1+R2,W/P1+R2,tv,0.45,0.776225637831183,0.0
coco,W/P1+R2,W/P1+R2,aeroplane,0.45,0.6338869758962933,0.0
coco,W/P1+R2,W/P1+R2,bicycle,0.45,0.5756120555512638,0.0
coco,W/P1+R2,W/P1+R2,bird,0.45,0.7558923622351892,0.0
coco,W/P1+R2,W/P1+R2,boat,0.45,0.5265864346345857,0.0
coco,W/P1+R2,W/P1+R2,bottle,0.45,0.47836333141167076,0.0
coco,W/P1+R2,W/P1+R2,bus,0.45,0.9059993461929059,0.0
coco,W/P1+R2,W/P1+R2,car,0.45,0.5583435351265461,0.0
coco,W/P1+R2,W/P1+R2,cat,0.45,0.8595094598967598,0.0
coco,W/P1+R2,W/P1+R2,chair,0.45,0.4503940821432896,0.0
coco,W/P1+R2,W/P1+R2,cow,0.45,0.8420356364682315,0.0
coco,W/P1+R2,W/P1+R2,dining table,0.45,0.36531745681201455,0.0
coco,W/P1+R2,W/P1+R2,dog,0.45,0.7561741373750404,0.0
coco,W/P1+R2,W/P1+R2,horse,0.45,0.6730046450362706,0.0
coco,W/P1+R2,W/P1+R2,motorbike,0.45,0.6291226698362047,0.0
coco,W/P1+R2,W/P1+R2,person,0.45,0.23138472474385038,0.0
coco,W/P1+R2,W/P1+R2,potted plant,0.45,0.6965014002178117,0.0
coco,W/P1+R2,W/P1+R2,sheep,0.45,0.8747748396858563,0.0
coco,W/P1+R2,W/P1+R2,sofa,0.45,0.6050312948498806,0.0
coco,W/P1+R2,W/P1+R2,train,0.45,0.6251210403413416,0.0
coco,W/P1+R2,W/P1+R2,tv,0.45,0.6997605334781958,0.0
voc,W/SG12/top50+R2,W/P1+R2,aeroplane,0.425,0.8875976083608393,-0.008623956546215616
voc,W/SG12/top50+R2,W/P1+R2,bicycle,0.425,0.7472476166777088,-0.015152636595163638
voc,W/SG12/top50+R2,W/P1+R2,bird,0.425,0.9008432218554098,0.017108614915690268
voc,W/SG12/top50+R2,W/P1+R2,boat,0.425,0.8226080541840253,-0.017046220484444374
voc,W/SG12/top50+R2,W/P1+R2,bottle,0.425,0.8804106728554493,0.003688803722447731
voc,W/SG12/top50+R2,W/P1+R2,bus,0.425,0.905788491982633,7.3495728016848005e-06
voc,W/SG12/top50+R2,W/P1+R2,car,0.425,0.8182266358534825,-0.008872985745797757
voc,W/SG12/top50+R2,W/P1+R2,cat,0.425,0.841684088811937,0.008582659451944474
voc,W/SG12/top50+R2,W/P1+R2,chair,0.425,0.7188347515509326,-0.05716012104158563
voc,W/SG12/top50+R2,W/P1+R2,cow,0.425,0.8862382622079279,0.030901823898164205
voc,W/SG12/top50+R2,W/P1+R2,dining-table,0.425,0.45029927227699784,-0.0001000097044493442
voc,W/SG12/top50+R2,W/P1+R2,dog,0.425,0.9264004543807421,0.011417204879299425
voc,W/SG12/top50+R2,W/P1+R2,horse,0.425,0.8989926155062556,0.003289914030804364
voc,W/SG12/top50+R2,W/P1+R2,motorbike,0.425,0.8356590050283362,-0.013005503311758426
voc,W/SG12/top50+R2,W/P1+R2,person,0.425,0.7602300699471132,0.02234993943403396
voc,W/SG12/top50+R2,W/P1+R2,potted-plant,0.425,0.755115482026844,0.0035159146101938488
voc,W/SG12/top50+R2,W/P1+R2,sheep,0.425,0.8282487082564177,0.01800597401375703
voc,W/SG12/top50+R2,W/P1+R2,sofa,0.425,0.9532758514672621,-0.0046771594781654935
voc,W/SG12/top50+R2,W/P1+R2,train,0.425,0.6465938902200664,-0.017884381578492703
voc,W/SG12/top50+R2,W/P1+R2,tv,0.425,0.7982218427283378,0.021996204897154814
coco,W/SG12/top50+R2,W/P1+R2,aeroplane,0.425,0.6139810937108154,-0.01990588218547784
coco,W/SG12/top50+R2,W/P1+R2,bicycle,0.425,0.5696029433721667,-0.006009112179097165
coco,W/SG12/top50+R2,W/P1+R2,bird,0.425,0.7410917580385353,-0.014800604196653855
coco,W/SG12/top50+R2,W/P1+R2,boat,0.425,0.518889736325997,-0.0076966983085886564
coco,W/SG12/top50+R2,W/P1+R2,bottle,0.425,0.46991946455757894,-0.00844386685409182
coco,W/SG12/top50+R2,W/P1+R2,bus,0.425,0.9052497768365266,-0.0007495693563792649
coco,W/SG12/top50+R2,W/P1+R2,car,0.425,0.5493665940508077,-0.008976941075738432
coco,W/SG12/top50+R2,W/P1+R2,cat,0.425,0.8514022511688242,-0.008107208727935578
coco,W/SG12/top50+R2,W/P1+R2,chair,0.425,0.4371469805993863,-0.013247101543903295
coco,W/SG12/top50+R2,W/P1+R2,cow,0.425,0.8497248089091591,0.007689172440927594
coco,W/SG12/top50+R2,W/P1+R2,dining table,0.425,0.3913854661423625,0.026068009330347974
coco,W/SG12/top50+R2,W/P1+R2,dog,0.425,0.7585571013138498,0.002382963938809435
coco,W/SG12/top50+R2,W/P1+R2,horse,0.425,0.6658591905393644,-0.007145454496906178
coco,W/SG12/top50+R2,W/P1+R2,motorbike,0.425,0.6208965420674353,-0.008226127768769409
coco,W/SG12/top50+R2,W/P1+R2,person,0.425,0.24434124127216175,0.012956516528311368
coco,W/SG12/top50+R2,W/P1+R2,potted plant,0.425,0.6790814865482933,-0.01741991366951845
coco,W/SG12/top50+R2,W/P1+R2,sheep,0.425,0.8798905581686302,0.005115718482773812
coco,W/SG12/top50+R2,W/P1+R2,sofa,0.425,0.5802818720029792,-0.024749422846901403
coco,W/SG12/top50+R2,W/P1+R2,train,0.425,0.6041976583617213,-0.020923381979620292
coco,W/SG12/top50+R2,W/P1+R2,tv,0.425,0.6991850304704748,-0.0005755030077210233
