# Locked sg14 results

VOC only selects. SG15 source thresholds selected on four training folds; refined outputs inherit them. Full-VOC choices frozen before COCO metrics are interpreted. Diagnostic variants never promoted on COCO.

Same-generation audited controls; historical cross-hardware discrepancies are diagnostic only. Does not certify historical bit-identical reproduction.

| Cell | VOC OOF mIoU | Full VOC threshold | Full VOC mIoU | COCO mIoU |
|---|---:|---:|---:|---:|
| W/REF | 0.787280 | 0.425 | 0.788610 | 0.617163 |
| W/SG11/trim20/rho0p80 | 0.798400 | 0.425 | 0.798400 | 0.621916 |
| W/SG12/top50 | 0.796658 | 0.450 | 0.798558 | 0.627174 |
| W/SG14 | 0.798651 | 0.450 | 0.798651 | 0.621678 |

## Protocol selection

| Screen | Winner | Votes | Selected OOF / REF | Gain | Frozen |
|---|---|---:|---:|---:|---|
| versus_W/REF | W/SG14 | 5/5 | 0.798651 / 0.787280 | +0.011371 | W/SG14 |
| versus_W/SG11/trim20/rho0p80 | W/SG11/trim20/rho0p80 | 5/5 | 0.798400 / 0.798400 | +0.000000 | W/SG11/trim20/rho0p80 |
| versus_W/SG12/top50 | W/SG12/top50 | 5/5 | 0.796658 / 0.796658 | +0.000000 | W/SG12/top50 |
| joint | W/SG11/trim20/rho0p80 | 5/5 | 0.798400 / 0.787280 | +0.011120 | W/SG11/trim20/rho0p80 |

Forced candidate OOF and protocol-selected OOF are distinct. All five fold winners, thresholds, class IoUs, deltas, paired 20k-bootstrap intervals and improved/worsened/tied counts are in summary.json and fold_results.json.

## Mechanism diagnostics

```json
{
  "voc": {
    "diagnostic_threshold": 0.45,
    "group_names": [
      "SG12_only",
      "SG11_only",
      "both_same_direction",
      "both_opposite_directions",
      "neither"
    ],
    "transition_methods": [
      "W/SG11/trim20/rho0p80",
      "W/SG12/top50",
      "W/SG14"
    ],
    "change_order": [
      "FN_to_TP",
      "FP_to_TN",
      "TP_to_FN",
      "TN_to_FP"
    ],
    "binary_group_changes_at_0p45": [
      [
        [
          0,
          0,
          0,
          0
        ],
        [
          526366,
          94913,
          57696,
          358330
        ],
        [
          404840,
          60372,
          24205,
          283764
        ]
      ],
      [
        [
          291325,
          857918,
          808537,
          180193
        ],
        [
          0,
          0,
          0,
          0
        ],
        [
          143537,
          159425,
          111110,
          71529
        ]
      ],
      [
        [
          294462,
          269414,
          57525,
          92370
        ],
        [
          294462,
          269414,
          57525,
          92370
        ],
        [
          280839,
          235474,
          51050,
          88609
        ]
      ],
      [
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ]
      ],
      [
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ],
        [
          177314,
          47743,
          42718,
          234014
        ]
      ]
    ],
    "binary_group_pixels_at0p45": [
      1037305,
      2137973,
      713771,
      0,
      124561511
    ],
    "continuous_group_changes_at0p45": [
      [
        [
          181,
          124,
          117,
          38
        ],
        [
          214,
          22,
          18,
          83
        ],
        [
          281,
          27,
          35,
          145
        ]
      ],
      [
        [
          87,
          44,
          40,
          25
        ],
        [
          53,
          13,
          4,
          16
        ],
        [
          82,
          22,
          22,
          45
        ]
      ],
      [
        [
          377738,
          656499,
          417489,
          160467
        ],
        [
          515468,
          268360,
          76614,
          260335
        ],
        [
          643348,
          353132,
          141811,
          411570
        ]
      ],
      [
        [
          207781,
          470665,
          448416,
          112032
        ],
        [
          305093,
          95932,
          38585,
          190265
        ],
        [
          362818,
          149833,
          87215,
          266155
        ]
      ],
      [
        [
          0,
          0,
          0,
          1
        ],
        [
          0,
          0,
          0,
          1
        ],
        [
          1,
          0,
          0,
          1
        ]
      ]
    ],
    "continuous_full_resolution_group_pixels": [
      10907,
      7059,
      102493603,
      25938989,
      2
    ],
    "continuous_raw_group_pixels": [
      150,
      95,
      1610331,
      396464,
      0
    ],
    "binary_opposite_direction_note": "Two binary predictions cannot move in opposite directions from the same binary REF; that group is exactly zero. Continuous raw-map directions are recorded separately.",
    "mean_consensus_fractions": [
      0.5,
      0.23520557637117348,
      0.14392189492984694,
      0.09998106664540816
    ],
    "consensus_fraction_order": [
      "observations_rejected",
      "all_observation_pixels_gate_rejected",
      "selected_observation_pixels_gate_rejected",
      "gate_membership_changed"
    ],
    "mean_per_image_reliability_correlations": [
      0.5070251117878535,
      -0.5070251122811225,
      0.6050320755401196,
      0.43953153096355596
    ],
    "correlation_order": [
      "raw_upward_on_GT_downward_on_BG_usefulness",
      "raw_harmfulness",
      "agreement_with_useful_REF_vs_CA_flips_at0p45",
      "agreement_with_harmful_REF_vs_CA_flips_at0p45"
    ],
    "all_threshold_group_counts_retained_in_sidecars": true,
    "group_changes_at_each_VOC_selected_threshold": {
      "W/SG11/trim20/rho0p80": [
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            419852,
            125347,
            58981,
            386224
          ],
          [
            326054,
            78936,
            23818,
            303090
          ]
        ],
        [
          [
            344863,
            828182,
            633174,
            256804
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            158704,
            166635,
            92260,
            99532
          ]
        ],
        [
          [
            273186,
            293131,
            43331,
            119833
          ],
          [
            273186,
            293131,
            43331,
            119833
          ],
          [
            258524,
            259881,
            37962,
            114821
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            130092,
            59890,
            40979,
            248563
          ]
        ]
      ],
      "W/SG12/top50": [
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            526366,
            94913,
            57696,
            358330
          ],
          [
            404840,
            60372,
            24205,
            283764
          ]
        ],
        [
          [
            291325,
            857918,
            808537,
            180193
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            143537,
            159425,
            111110,
            71529
          ]
        ],
        [
          [
            294462,
            269414,
            57525,
            92370
          ],
          [
            294462,
            269414,
            57525,
            92370
          ],
          [
            280839,
            235474,
            51050,
            88609
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            177314,
            47743,
            42718,
            234014
          ]
        ]
      ],
      "W/SG14": [
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            526366,
            94913,
            57696,
            358330
          ],
          [
            404840,
            60372,
            24205,
            283764
          ]
        ],
        [
          [
            291325,
            857918,
            808537,
            180193
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            143537,
            159425,
            111110,
            71529
          ]
        ],
        [
          [
            294462,
            269414,
            57525,
            92370
          ],
          [
            294462,
            269414,
            57525,
            92370
          ],
          [
            280839,
            235474,
            51050,
            88609
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            177314,
            47743,
            42718,
            234014
          ]
        ]
      ]
    }
  },
  "coco": {
    "diagnostic_threshold": 0.45,
    "group_names": [
      "SG12_only",
      "SG11_only",
      "both_same_direction",
      "both_opposite_directions",
      "neither"
    ],
    "transition_methods": [
      "W/SG11/trim20/rho0p80",
      "W/SG12/top50",
      "W/SG14"
    ],
    "change_order": [
      "FN_to_TP",
      "FP_to_TN",
      "TP_to_FN",
      "TN_to_FP"
    ],
    "binary_group_changes_at_0p45": [
      [
        [
          0,
          0,
          0,
          0
        ],
        [
          662353,
          113992,
          35850,
          1239641
        ],
        [
          460280,
          76096,
          20406,
          981396
        ]
      ],
      [
        [
          132445,
          2065193,
          1170992,
          568744
        ],
        [
          0,
          0,
          0,
          0
        ],
        [
          68320,
          271488,
          131817,
          217808
        ]
      ],
      [
        [
          305630,
          353395,
          75198,
          646569
        ],
        [
          305630,
          353395,
          75198,
          646569
        ],
        [
          290487,
          304539,
          64260,
          603899
        ]
      ],
      [
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ]
      ],
      [
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ],
        [
          155984,
          103917,
          36022,
          752100
        ]
      ]
    ],
    "binary_group_pixels_at0p45": [
      2051836,
      3937374,
      1380792,
      0,
      177965806
    ],
    "continuous_group_changes_at0p45": [
      [
        [
          87,
          385,
          228,
          305
        ],
        [
          272,
          37,
          8,
          448
        ],
        [
          240,
          81,
          48,
          685
        ]
      ],
      [
        [
          47,
          164,
          92,
          141
        ],
        [
          71,
          35,
          6,
          137
        ],
        [
          66,
          48,
          25,
          207
        ]
      ],
      [
        [
          300988,
          1433848,
          674696,
          808395
        ],
        [
          577127,
          346162,
          80008,
          1154618
        ],
        [
          629822,
          528166,
          159400,
          1613538
        ]
      ],
      [
        [
          136953,
          984191,
          571174,
          406472
        ],
        [
          390513,
          121153,
          31026,
          731007
        ],
        [
          344943,
          227745,
          93032,
          940773
        ]
      ],
      [
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0
        ]
      ]
    ],
    "continuous_full_resolution_group_pixels": [
      25664,
      15866,
      142320162,
      42974111,
      5
    ],
    "continuous_raw_group_pixels": [
      384,
      227,
      2247265,
      647996,
      0
    ],
    "binary_opposite_direction_note": "Two binary predictions cannot move in opposite directions from the same binary REF; that group is exactly zero. Continuous raw-map directions are recorded separately.",
    "mean_consensus_fractions": [
      0.5,
      0.21887707743988685,
      0.13859763138702264,
      0.08973704638879067
    ],
    "consensus_fraction_order": [
      "observations_rejected",
      "all_observation_pixels_gate_rejected",
      "selected_observation_pixels_gate_rejected",
      "gate_membership_changed"
    ],
    "mean_per_image_reliability_correlations": [
      0.30554010371376544,
      -0.30554010339918564,
      0.5893022127631853,
      0.5316468597686761
    ],
    "correlation_order": [
      "raw_upward_on_GT_downward_on_BG_usefulness",
      "raw_harmfulness",
      "agreement_with_useful_REF_vs_CA_flips_at0p45",
      "agreement_with_harmful_REF_vs_CA_flips_at0p45"
    ],
    "all_threshold_group_counts_retained_in_sidecars": true,
    "group_changes_at_each_VOC_selected_threshold": {
      "W/SG11/trim20/rho0p80": [
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            527475,
            133534,
            34805,
            1277383
          ],
          [
            386894,
            86889,
            20482,
            1039386
          ]
        ],
        [
          [
            148498,
            2034643,
            952273,
            679226
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            72521,
            281750,
            112591,
            259251
          ]
        ],
        [
          [
            267352,
            381512,
            60073,
            669278
          ],
          [
            267352,
            381512,
            60073,
            669278
          ],
          [
            250488,
            320658,
            52338,
            631544
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            132579,
            97013,
            34712,
            800922
          ]
        ]
      ],
      "W/SG12/top50": [
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            662353,
            113992,
            35850,
            1239641
          ],
          [
            460280,
            76096,
            20406,
            981396
          ]
        ],
        [
          [
            132445,
            2065193,
            1170992,
            568744
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            68320,
            271488,
            131817,
            217808
          ]
        ],
        [
          [
            305630,
            353395,
            75198,
            646569
          ],
          [
            305630,
            353395,
            75198,
            646569
          ],
          [
            290487,
            304539,
            64260,
            603899
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            155984,
            103917,
            36022,
            752100
          ]
        ]
      ],
      "W/SG14": [
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            662353,
            113992,
            35850,
            1239641
          ],
          [
            460280,
            76096,
            20406,
            981396
          ]
        ],
        [
          [
            132445,
            2065193,
            1170992,
            568744
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            68320,
            271488,
            131817,
            217808
          ]
        ],
        [
          [
            305630,
            353395,
            75198,
            646569
          ],
          [
            305630,
            353395,
            75198,
            646569
          ],
          [
            290487,
            304539,
            64260,
            603899
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        [
          [
            0,
            0,
            0,
            0
          ],
          [
            0,
            0,
            0,
            0
          ],
          [
            155984,
            103917,
            36022,
            752100
          ]
        ]
      ]
    }
  }
}
```

## Five folds and forced OOF comparisons

```json
{
  "screens": {
    "versus_W/REF": {
      "reference": "W/REF",
      "candidates": [
        "W/SG14"
      ],
      "folds": [
        {
          "fold": 0,
          "chosen": "W/SG14",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.425,
          "reason": "conditional",
          "train_miou_chosen": 0.7985089720831644,
          "train_miou_reference": 0.7899826822564588,
          "heldout_miou_chosen": 0.8016309876688732,
          "heldout_miou_reference": 0.7854325094473272
        },
        {
          "fold": 1,
          "chosen": "W/SG14",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.425,
          "reason": "promote",
          "train_miou_chosen": 0.7933510901389164,
          "train_miou_reference": 0.7824392230693771,
          "heldout_miou_chosen": 0.8182065636280397,
          "heldout_miou_reference": 0.8114630875761657
        },
        {
          "fold": 2,
          "chosen": "W/SG14",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.425,
          "reason": "conditional",
          "train_miou_chosen": 0.7999088545750657,
          "train_miou_reference": 0.7900353499208056,
          "heldout_miou_chosen": 0.7972752556897104,
          "heldout_miou_reference": 0.7859289428172378
        },
        {
          "fold": 3,
          "chosen": "W/SG14",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.425,
          "reason": "conditional",
          "train_miou_chosen": 0.800265610621889,
          "train_miou_reference": 0.7910934555087368,
          "heldout_miou_chosen": 0.7952078536912044,
          "heldout_miou_reference": 0.7802224546432177
        },
        {
          "fold": 4,
          "chosen": "W/SG14",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "promote",
          "train_miou_chosen": 0.8013598105780046,
          "train_miou_reference": 0.7908405743685762,
          "heldout_miou_chosen": 0.7840861190218947,
          "heldout_miou_reference": 0.7751596960962914
        }
      ],
      "selection_counts": {
        "W/SG14": 5
      },
      "vote_winner": "W/SG14",
      "vote_winner_votes": 5,
      "frozen": "W/SG14",
      "freeze_reason": "fold vote and OOF guard passed",
      "oof_miou_selection": 0.7986510455041298,
      "oof_miou_reference": 0.7872797008862745,
      "oof_gain": 0.011371344617855272,
      "full_threshold_frozen": 0.45,
      "full_threshold_reference": 0.425,
      "full_miou_frozen": 0.7986510455041298,
      "full_miou_reference": 0.7886099033007808,
      "paired_oof": {
        "mean": 0.012898143558277928,
        "ci_low": 0.009457659981610743,
        "ci_high": 0.016345317892649758,
        "better": 339,
        "worse": 151,
        "ties": 0
      }
    },
    "versus_W/SG11/trim20/rho0p80": {
      "reference": "W/SG11/trim20/rho0p80",
      "candidates": [
        "W/SG14"
      ],
      "folds": [
        {
          "fold": 0,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.7988334127115324,
          "train_miou_reference": 0.7988334127115324,
          "heldout_miou_chosen": 0.7981035044997601,
          "heldout_miou_reference": 0.7981035044997601
        },
        {
          "fold": 1,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.7938446893296913,
          "train_miou_reference": 0.7938446893296913,
          "heldout_miou_chosen": 0.8152177748924299,
          "heldout_miou_reference": 0.8152177748924299
        },
        {
          "fold": 2,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.798978351866665,
          "train_miou_reference": 0.798978351866665,
          "heldout_miou_chosen": 0.7983786747748505,
          "heldout_miou_reference": 0.7983786747748505
        },
        {
          "fold": 3,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.7999777499621435,
          "train_miou_reference": 0.7999777499621435,
          "heldout_miou_chosen": 0.7925278595866733,
          "heldout_miou_reference": 0.7925278595866733
        },
        {
          "fold": 4,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8001730637613649,
          "train_miou_reference": 0.8001730637613649,
          "heldout_miou_chosen": 0.7859481776989832,
          "heldout_miou_reference": 0.7859481776989832
        }
      ],
      "selection_counts": {
        "W/SG11/trim20/rho0p80": 5
      },
      "vote_winner": "W/SG11/trim20/rho0p80",
      "vote_winner_votes": 5,
      "frozen": "W/SG11/trim20/rho0p80",
      "freeze_reason": "reference won the fold vote",
      "oof_miou_selection": 0.798400027953512,
      "oof_miou_reference": 0.798400027953512,
      "oof_gain": 0.0,
      "full_threshold_frozen": 0.425,
      "full_threshold_reference": 0.425,
      "full_miou_frozen": 0.798400027953512,
      "full_miou_reference": 0.798400027953512,
      "paired_oof": {
        "mean": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "better": 0,
        "worse": 0,
        "ties": 490
      }
    },
    "versus_W/SG12/top50": {
      "reference": "W/SG12/top50",
      "candidates": [
        "W/SG14"
      ],
      "folds": [
        {
          "fold": 0,
          "chosen": "W/SG12/top50",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.7993483891359897,
          "train_miou_reference": 0.7993483891359897,
          "heldout_miou_chosen": 0.7980079201556065,
          "heldout_miou_reference": 0.7980079201556065
        },
        {
          "fold": 1,
          "chosen": "W/SG12/top50",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.792573796601294,
          "train_miou_reference": 0.792573796601294,
          "heldout_miou_chosen": 0.8210694391882276,
          "heldout_miou_reference": 0.8210694391882276
        },
        {
          "fold": 2,
          "chosen": "W/SG12/top50",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8000770040354752,
          "train_miou_reference": 0.8000770040354752,
          "heldout_miou_chosen": 0.7957113805320305,
          "heldout_miou_reference": 0.7957113805320305
        },
        {
          "fold": 3,
          "chosen": "W/SG12/top50",
          "chosen_threshold": 0.45,
          "reference_threshold": 0.45,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8011205136263397,
          "train_miou_reference": 0.8011205136263397,
          "heldout_miou_chosen": 0.7904002140640006,
          "heldout_miou_reference": 0.7904002140640006
        },
        {
          "fold": 4,
          "chosen": "W/SG12/top50",
          "chosen_threshold": 0.475,
          "reference_threshold": 0.475,
          "reason": "no eligible candidate",
          "train_miou_chosen": 0.8000199383633966,
          "train_miou_reference": 0.8000199383633966,
          "heldout_miou_chosen": 0.7811817777136335,
          "heldout_miou_reference": 0.7811817777136335
        }
      ],
      "selection_counts": {
        "W/SG12/top50": 5
      },
      "vote_winner": "W/SG12/top50",
      "vote_winner_votes": 5,
      "frozen": "W/SG12/top50",
      "freeze_reason": "reference won the fold vote",
      "oof_miou_selection": 0.796657782422708,
      "oof_miou_reference": 0.796657782422708,
      "oof_gain": 0.0,
      "full_threshold_frozen": 0.45,
      "full_threshold_reference": 0.45,
      "full_miou_frozen": 0.7985584168680424,
      "full_miou_reference": 0.7985584168680424,
      "paired_oof": {
        "mean": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "better": 0,
        "worse": 0,
        "ties": 490
      }
    },
    "joint": {
      "reference": "W/REF",
      "candidates": [
        "W/SG11/trim20/rho0p80",
        "W/SG12/top50",
        "W/SG14"
      ],
      "folds": [
        {
          "fold": 0,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "conditional",
          "train_miou_chosen": 0.7988334127115324,
          "train_miou_reference": 0.7899826822564588,
          "heldout_miou_chosen": 0.7981035044997601,
          "heldout_miou_reference": 0.7854325094473272
        },
        {
          "fold": 1,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "promote",
          "train_miou_chosen": 0.7938446893296913,
          "train_miou_reference": 0.7824392230693771,
          "heldout_miou_chosen": 0.8152177748924299,
          "heldout_miou_reference": 0.8114630875761657
        },
        {
          "fold": 2,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "conditional",
          "train_miou_chosen": 0.798978351866665,
          "train_miou_reference": 0.7900353499208056,
          "heldout_miou_chosen": 0.7983786747748505,
          "heldout_miou_reference": 0.7859289428172378
        },
        {
          "fold": 3,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.425,
          "reason": "conditional",
          "train_miou_chosen": 0.7999777499621435,
          "train_miou_reference": 0.7910934555087368,
          "heldout_miou_chosen": 0.7925278595866733,
          "heldout_miou_reference": 0.7802224546432177
        },
        {
          "fold": 4,
          "chosen": "W/SG11/trim20/rho0p80",
          "chosen_threshold": 0.425,
          "reference_threshold": 0.45,
          "reason": "conditional",
          "train_miou_chosen": 0.8001730637613649,
          "train_miou_reference": 0.7908405743685762,
          "heldout_miou_chosen": 0.7859481776989832,
          "heldout_miou_reference": 0.7751596960962914
        }
      ],
      "selection_counts": {
        "W/SG11/trim20/rho0p80": 5
      },
      "vote_winner": "W/SG11/trim20/rho0p80",
      "vote_winner_votes": 5,
      "frozen": "W/SG11/trim20/rho0p80",
      "freeze_reason": "fold vote and OOF guard passed",
      "oof_miou_selection": 0.798400027953512,
      "oof_miou_reference": 0.7872797008862745,
      "oof_gain": 0.011120327067237423,
      "full_threshold_frozen": 0.425,
      "full_threshold_reference": 0.425,
      "full_miou_frozen": 0.798400027953512,
      "full_miou_reference": 0.7886099033007808,
      "paired_oof": {
        "mean": 0.009201382470732647,
        "ci_low": 0.005272862849234696,
        "ci_high": 0.013165300665173005,
        "better": 276,
        "worse": 214,
        "ties": 0
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
    "W/SG11/trim20/rho0p80": {
      "miou": 0.798400027953512,
      "folds": [
        {
          "fold": 0,
          "threshold": 0.425,
          "heldout_miou": 0.7981035044997601
        },
        {
          "fold": 1,
          "threshold": 0.425,
          "heldout_miou": 0.8152177748924299
        },
        {
          "fold": 2,
          "threshold": 0.425,
          "heldout_miou": 0.7983786747748505
        },
        {
          "fold": 3,
          "threshold": 0.425,
          "heldout_miou": 0.7925278595866733
        },
        {
          "fold": 4,
          "threshold": 0.425,
          "heldout_miou": 0.7859481776989832
        }
      ],
      "mean_image_iou": 0.7974127767071819
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
    "W/SG14": {
      "miou": 0.7986510455041298,
      "folds": [
        {
          "fold": 0,
          "threshold": 0.45,
          "heldout_miou": 0.8016309876688732
        },
        {
          "fold": 1,
          "threshold": 0.45,
          "heldout_miou": 0.8182065636280397
        },
        {
          "fold": 2,
          "threshold": 0.45,
          "heldout_miou": 0.7972752556897104
        },
        {
          "fold": 3,
          "threshold": 0.45,
          "heldout_miou": 0.7952078536912044
        },
        {
          "fold": 4,
          "threshold": 0.45,
          "heldout_miou": 0.7840861190218947
        }
      ],
      "mean_image_iou": 0.8011095377947273
    }
  },
  "forced_oof_comparisons": {
    "versus_W/REF": {
      "W/SG14": {
        "candidate_miou": 0.7986510455041298,
        "reference_miou": 0.7872797008862745,
        "delta": 0.011371344617855272,
        "paired_mean_image_iou": {
          "mean": 0.012898143558277928,
          "ci_low": 0.009457659981610743,
          "ci_high": 0.016345317892649758,
          "better": 339,
          "worse": 151,
          "ties": 0
        }
      }
    },
    "versus_W/SG11/trim20/rho0p80": {
      "W/SG14": {
        "candidate_miou": 0.7986510455041298,
        "reference_miou": 0.798400027953512,
        "delta": 0.00025101755061784914,
        "paired_mean_image_iou": {
          "mean": 0.0036967610875452807,
          "ci_low": 0.00013832783538174096,
          "ci_high": 0.007371945530313301,
          "better": 304,
          "worse": 186,
          "ties": 0
        }
      }
    },
    "versus_W/SG12/top50": {
      "W/SG14": {
        "candidate_miou": 0.7986510455041298,
        "reference_miou": 0.796657782422708,
        "delta": 0.001993263081421781,
        "paired_mean_image_iou": {
          "mean": 0.0018422037282736618,
          "ci_low": -0.0003803862190036475,
          "ci_high": 0.0041345858581175185,
          "better": 242,
          "worse": 248,
          "ties": 0
        }
      }
    },
    "joint": {
      "W/SG11/trim20/rho0p80": {
        "candidate_miou": 0.798400027953512,
        "reference_miou": 0.7872797008862745,
        "delta": 0.011120327067237423,
        "paired_mean_image_iou": {
          "mean": 0.009201382470732647,
          "ci_low": 0.005272862849234696,
          "ci_high": 0.013165300665173005,
          "better": 276,
          "worse": 214,
          "ties": 0
        }
      },
      "W/SG12/top50": {
        "candidate_miou": 0.796657782422708,
        "reference_miou": 0.7872797008862745,
        "delta": 0.009378081536433491,
        "paired_mean_image_iou": {
          "mean": 0.011055939830004266,
          "ci_low": 0.008415566487768339,
          "ci_high": 0.013674811070246182,
          "better": 371,
          "worse": 119,
          "ties": 0
        }
      },
      "W/SG14": {
        "candidate_miou": 0.7986510455041298,
        "reference_miou": 0.7872797008862745,
        "delta": 0.011371344617855272,
        "paired_mean_image_iou": {
          "mean": 0.012898143558277928,
          "ci_low": 0.009457659981610743,
          "ci_high": 0.016345317892649758,
          "better": 339,
          "worse": 151,
          "ties": 0
        }
      }
    }
  },
  "screen_frozen_transfer": {
    "versus_W/REF": {
      "reference": "W/REF",
      "frozen": "W/SG14",
      "reference_threshold": 0.425,
      "frozen_threshold": 0.45,
      "reference_coco_miou": 0.6171634284640197,
      "frozen_coco_miou": 0.6216777775615738
    },
    "versus_W/SG11/trim20/rho0p80": {
      "reference": "W/SG11/trim20/rho0p80",
      "frozen": "W/SG11/trim20/rho0p80",
      "reference_threshold": 0.425,
      "frozen_threshold": 0.425,
      "reference_coco_miou": 0.6219155223421355,
      "frozen_coco_miou": 0.6219155223421355
    },
    "versus_W/SG12/top50": {
      "reference": "W/SG12/top50",
      "frozen": "W/SG12/top50",
      "reference_threshold": 0.45,
      "frozen_threshold": 0.45,
      "reference_coco_miou": 0.6271735100277616,
      "frozen_coco_miou": 0.6271735100277616
    },
    "joint": {
      "reference": "W/REF",
      "frozen": "W/SG11/trim20/rho0p80",
      "reference_threshold": 0.425,
      "frozen_threshold": 0.425,
      "reference_coco_miou": 0.6171634284640197,
      "frozen_coco_miou": 0.6219155223421355
    }
  }
}
```

## All class results

dataset,cell,reference,class,threshold,iou,delta
voc,W/REF,W/REF,aeroplane,0.425,0.8352635722287826,0.0
voc,W/REF,W/REF,bicycle,0.425,0.737438165800942,0.0
voc,W/REF,W/REF,bird,0.425,0.8403247874482437,0.0
voc,W/REF,W/REF,boat,0.425,0.8124661032715629,0.0
voc,W/REF,W/REF,bottle,0.425,0.8561283615303112,0.0
voc,W/REF,W/REF,bus,0.425,0.9089213701956407,0.0
voc,W/REF,W/REF,car,0.425,0.8254276616207066,0.0
voc,W/REF,W/REF,cat,0.425,0.8131546400743833,0.0
voc,W/REF,W/REF,chair,0.425,0.629114721949853,0.0
voc,W/REF,W/REF,cow,0.425,0.8426602356685133,0.0
voc,W/REF,W/REF,dining-table,0.425,0.45270798975063226,0.0
voc,W/REF,W/REF,dog,0.425,0.9028583590477207,0.0
voc,W/REF,W/REF,horse,0.425,0.8655564550590958,0.0
voc,W/REF,W/REF,motorbike,0.425,0.839833950438049,0.0
voc,W/REF,W/REF,person,0.425,0.7057082886035627,0.0
voc,W/REF,W/REF,potted-plant,0.425,0.7310806715606982,0.0
voc,W/REF,W/REF,sheep,0.425,0.8001454640174295,0.0
voc,W/REF,W/REF,sofa,0.425,0.9368351592460529,0.0
voc,W/REF,W/REF,train,0.425,0.6433302895856994,0.0
voc,W/REF,W/REF,tv,0.425,0.7932418189177365,0.0
coco,W/REF,W/REF,aeroplane,0.425,0.5639089672172332,0.0
coco,W/REF,W/REF,bicycle,0.425,0.5813415797451317,0.0
coco,W/REF,W/REF,bird,0.425,0.7128468945430074,0.0
coco,W/REF,W/REF,boat,0.425,0.5174657041725186,0.0
coco,W/REF,W/REF,bottle,0.425,0.45657259822166135,0.0
coco,W/REF,W/REF,bus,0.425,0.9200937035017466,0.0
coco,W/REF,W/REF,car,0.425,0.5682754030772628,0.0
coco,W/REF,W/REF,cat,0.425,0.8286793983910459,0.0
coco,W/REF,W/REF,chair,0.425,0.4190301468853303,0.0
coco,W/REF,W/REF,cow,0.425,0.8005908527741239,0.0
coco,W/REF,W/REF,dining table,0.425,0.3742098228693574,0.0
coco,W/REF,W/REF,dog,0.425,0.7450744893666833,0.0
coco,W/REF,W/REF,horse,0.425,0.6354861139057686,0.0
coco,W/REF,W/REF,motorbike,0.425,0.6175213013121482,0.0
coco,W/REF,W/REF,person,0.425,0.21896664030149351,0.0
coco,W/REF,W/REF,potted plant,0.425,0.6430209450112684,0.0
coco,W/REF,W/REF,sheep,0.425,0.8455591644890148,0.0
coco,W/REF,W/REF,sofa,0.425,0.5901932607758212,0.0
coco,W/REF,W/REF,train,0.425,0.6075123829233591,0.0
coco,W/REF,W/REF,tv,0.425,0.6969191997964193,0.0
voc,W/SG11/trim20/rho0p80,W/REF,aeroplane,0.425,0.8464271651296567,0.011163592900874098
voc,W/SG11/trim20/rho0p80,W/REF,bicycle,0.425,0.7297935426619409,-0.007644623139001161
voc,W/SG11/trim20/rho0p80,W/REF,bird,0.425,0.8455293454533959,0.005204558005152182
voc,W/SG11/trim20/rho0p80,W/REF,boat,0.425,0.8452502193137245,0.03278411604216158
voc,W/SG11/trim20/rho0p80,W/REF,bottle,0.425,0.8664388294087437,0.010310467878432483
voc,W/SG11/trim20/rho0p80,W/REF,bus,0.425,0.9188366125574761,0.009915242361835386
voc,W/SG11/trim20/rho0p80,W/REF,car,0.425,0.826119225745161,0.0006915641244543691
voc,W/SG11/trim20/rho0p80,W/REF,cat,0.425,0.8276764835115531,0.014521843437169779
voc,W/SG11/trim20/rho0p80,W/REF,chair,0.425,0.6767276701797368,0.04761294822988382
voc,W/SG11/trim20/rho0p80,W/REF,cow,0.425,0.8484772420808906,0.005817006412377257
voc,W/SG11/trim20/rho0p80,W/REF,dining-table,0.425,0.46858883153663017,0.015880841785997912
voc,W/SG11/trim20/rho0p80,W/REF,dog,0.425,0.9012782994867968,-0.0015800595609238854
voc,W/SG11/trim20/rho0p80,W/REF,horse,0.425,0.8677548455394017,0.0021983904803059007
voc,W/SG11/trim20/rho0p80,W/REF,motorbike,0.425,0.8352628849213675,-0.004571065516681516
voc,W/SG11/trim20/rho0p80,W/REF,person,0.425,0.6971892634193377,-0.008519025184224982
voc,W/SG11/trim20/rho0p80,W/REF,potted-plant,0.425,0.7212579012695256,-0.009822770291172622
voc,W/SG11/trim20/rho0p80,W/REF,sheep,0.425,0.8409600522120914,0.04081458819466188
voc,W/SG11/trim20/rho0p80,W/REF,sofa,0.425,0.9309212897438567,-0.0059138695021961896
voc,W/SG11/trim20/rho0p80,W/REF,train,0.425,0.6795840740946458,0.03625378450894645
voc,W/SG11/trim20/rho0p80,W/REF,tv,0.425,0.7939267808043061,0.000684961886569635
coco,W/SG11/trim20/rho0p80,W/REF,aeroplane,0.425,0.5938088895748422,0.029899922357609077
coco,W/SG11/trim20/rho0p80,W/REF,bicycle,0.425,0.5970640900165662,0.015722510271434564
coco,W/SG11/trim20/rho0p80,W/REF,bird,0.425,0.7056429928181055,-0.007203901724901929
coco,W/SG11/trim20/rho0p80,W/REF,boat,0.425,0.5490289328595861,0.03156322868706751
coco,W/SG11/trim20/rho0p80,W/REF,bottle,0.425,0.4831149311523209,0.02654233293065955
coco,W/SG11/trim20/rho0p80,W/REF,bus,0.425,0.9242249786981986,0.004131275196452067
coco,W/SG11/trim20/rho0p80,W/REF,car,0.425,0.5465319265416768,-0.02174347653558595
coco,W/SG11/trim20/rho0p80,W/REF,cat,0.425,0.8377221168691091,0.00904271847806326
coco,W/SG11/trim20/rho0p80,W/REF,chair,0.425,0.4341028152434696,0.015072668358139307
coco,W/SG11/trim20/rho0p80,W/REF,cow,0.425,0.8124352985564612,0.011844445782337343
coco,W/SG11/trim20/rho0p80,W/REF,dining table,0.425,0.33358330756204163,-0.04062651530731576
coco,W/SG11/trim20/rho0p80,W/REF,dog,0.425,0.7435615415477532,-0.0015129478189300771
coco,W/SG11/trim20/rho0p80,W/REF,horse,0.425,0.6520243605775446,0.016538246671776058
coco,W/SG11/trim20/rho0p80,W/REF,motorbike,0.425,0.6293290066631625,0.011807705351014275
coco,W/SG11/trim20/rho0p80,W/REF,person,0.425,0.22128343391879743,0.002316793617303914
coco,W/SG11/trim20/rho0p80,W/REF,potted plant,0.425,0.6548142760133022,0.011793331002033836
coco,W/SG11/trim20/rho0p80,W/REF,sheep,0.425,0.8616753229243954,0.016116158435380612
coco,W/SG11/trim20/rho0p80,W/REF,sofa,0.425,0.570680239796403,-0.019513020979418205
coco,W/SG11/trim20/rho0p80,W/REF,train,0.425,0.6026238821184169,-0.004888500804942186
coco,W/SG11/trim20/rho0p80,W/REF,tv,0.425,0.68505810339056,-0.011861096405859306
voc,W/SG12/top50,W/REF,aeroplane,0.45,0.8551831673256742,0.019919595096891674
voc,W/SG12/top50,W/REF,bicycle,0.45,0.7452900929578082,0.007851927156866223
voc,W/SG12/top50,W/REF,bird,0.45,0.8564873864040521,0.016162598955808405
voc,W/SG12/top50,W/REF,boat,0.45,0.8349081845199188,0.022442081248355894
voc,W/SG12/top50,W/REF,bottle,0.45,0.8650743004251582,0.008945938894846961
voc,W/SG12/top50,W/REF,bus,0.45,0.9149153182349083,0.005993948039267671
voc,W/SG12/top50,W/REF,car,0.45,0.8331636144818277,0.007735952861121165
voc,W/SG12/top50,W/REF,cat,0.45,0.8088836400489084,-0.004271000025474869
voc,W/SG12/top50,W/REF,chair,0.45,0.6774630278879507,0.04834830593809769
voc,W/SG12/top50,W/REF,cow,0.45,0.8456211706998886,0.002960935031375289
voc,W/SG12/top50,W/REF,dining-table,0.45,0.4615386029282244,0.00883061317759215
voc,W/SG12/top50,W/REF,dog,0.45,0.9021113160917039,-0.0007470429560167968
voc,W/SG12/top50,W/REF,horse,0.45,0.8775036538322383,0.011947198773142431
voc,W/SG12/top50,W/REF,motorbike,0.45,0.8500637872967285,0.010229836858679509
voc,W/SG12/top50,W/REF,person,0.45,0.7225903577327976,0.016882069129234845
voc,W/SG12/top50,W/REF,potted-plant,0.45,0.7340156136998037,0.002934942139105501
voc,W/SG12/top50,W/REF,sheep,0.45,0.7844944353009128,-0.01565102871651669
voc,W/SG12/top50,W/REF,sofa,0.45,0.9477164299458083,0.010881270699755419
voc,W/SG12/top50,W/REF,train,0.45,0.669341712707643,0.0260114231219436
voc,W/SG12/top50,W/REF,tv,0.45,0.7848025248388913,-0.008439294078845139
coco,W/SG12/top50,W/REF,aeroplane,0.45,0.5900271553321144,0.026118188114881202
coco,W/SG12/top50,W/REF,bicycle,0.45,0.5803237423280837,-0.0010178374170479731
coco,W/SG12/top50,W/REF,bird,0.45,0.7159952205631942,0.0031483260201867758
coco,W/SG12/top50,W/REF,boat,0.45,0.530093329622414,0.012627625449895374
coco,W/SG12/top50,W/REF,bottle,0.45,0.4730029835566697,0.016430385335008346
coco,W/SG12/top50,W/REF,bus,0.45,0.9178582139706918,-0.0022354895310547906
coco,W/SG12/top50,W/REF,car,0.45,0.5629056930097877,-0.005369710067475109
coco,W/SG12/top50,W/REF,cat,0.45,0.8387175174664953,0.010038119075449381
coco,W/SG12/top50,W/REF,chair,0.45,0.42745203301870277,0.00842188613337247
coco,W/SG12/top50,W/REF,cow,0.45,0.8159134906457183,0.01532263787159438
coco,W/SG12/top50,W/REF,dining table,0.45,0.3807444032975091,0.006534580428151693
coco,W/SG12/top50,W/REF,dog,0.45,0.7460987487672033,0.0010242594005199779
coco,W/SG12/top50,W/REF,horse,0.45,0.6540135143737305,0.018527400467961863
coco,W/SG12/top50,W/REF,motorbike,0.45,0.6275318139701992,0.010010512658050974
coco,W/SG12/top50,W/REF,person,0.45,0.23774273358530068,0.018776093283807166
coco,W/SG12/top50,W/REF,potted plant,0.45,0.660547683027553,0.01752673801628457
coco,W/SG12/top50,W/REF,sheep,0.45,0.8677052584529141,0.022146093963899327
coco,W/SG12/top50,W/REF,sofa,0.45,0.5984718293318083,0.008278568555987076
coco,W/SG12/top50,W/REF,train,0.45,0.6181185621875782,0.01060617926421914
coco,W/SG12/top50,W/REF,tv,0.45,0.7002062740475652,0.0032870742511459383
voc,W/SG14,W/REF,aeroplane,0.45,0.8574259334632934,0.022162361234510874
voc,W/SG14,W/REF,bicycle,0.45,0.7413855485196968,0.003947382718754833
voc,W/SG14,W/REF,bird,0.45,0.8473770202581942,0.007052232809950487
voc,W/SG14,W/REF,boat,0.45,0.8364756364111381,0.024009533139575212
voc,W/SG14,W/REF,bottle,0.45,0.8579356157661583,0.0018072542358470445
voc,W/SG14,W/REF,bus,0.45,0.9145717851005302,0.005650414904889489
voc,W/SG14,W/REF,car,0.45,0.8281143027047619,0.0026866410840553634
voc,W/SG14,W/REF,cat,0.45,0.8094745289142482,-0.0036801111601351133
voc,W/SG14,W/REF,chair,0.45,0.6911345217679556,0.06201979981810268
voc,W/SG14,W/REF,cow,0.45,0.8528228373188482,0.01016260165033489
voc,W/SG14,W/REF,dining-table,0.45,0.4517079429752259,-0.0010000467754063824
voc,W/SG14,W/REF,dog,0.45,0.8985127805562464,-0.004345578491474322
voc,W/SG14,W/REF,horse,0.45,0.8753672023834825,0.009810747324386693
voc,W/SG14,W/REF,motorbike,0.45,0.84819655243237,0.00836260199432104
voc,W/SG14,W/REF,person,0.45,0.7105323594331052,0.004824070829542504
voc,W/SG14,W/REF,potted-plant,0.45,0.7372653156994998,0.006184644138801598
voc,W/SG14,W/REF,sheep,0.45,0.8237350955176377,0.023589631500208208
voc,W/SG14,W/REF,sofa,0.45,0.9423817423882767,0.005546583142223782
voc,W/SG14,W/REF,train,0.45,0.6712633010828319,0.027933011497132565
voc,W/SG14,W/REF,tv,0.45,0.7773408873890955,-0.015900931528641027
coco,W/SG14,W/REF,aeroplane,0.45,0.5857676340610951,0.02185866684386195
coco,W/SG14,W/REF,bicycle,0.45,0.5774944373005313,-0.003847142444600382
coco,W/SG14,W/REF,bird,0.45,0.7134946717511566,0.0006477772081491251
coco,W/SG14,W/REF,boat,0.45,0.528870552151554,0.011404847979035404
coco,W/SG14,W/REF,bottle,0.45,0.47724205095597794,0.020669452734316585
coco,W/SG14,W/REF,bus,0.45,0.9146018122163311,-0.0054918912854154955
coco,W/SG14,W/REF,car,0.45,0.5527238290241858,-0.015551574053076944
coco,W/SG14,W/REF,cat,0.45,0.843129736538289,0.014450338147243169
coco,W/SG14,W/REF,chair,0.45,0.4340217426440786,0.01499159575874831
coco,W/SG14,W/REF,cow,0.45,0.815973398813835,0.01538254603971112
coco,W/SG14,W/REF,dining table,0.45,0.37417748979733767,-3.233307201971902e-05
coco,W/SG14,W/REF,dog,0.45,0.7456205204719144,0.000546031105231104
coco,W/SG14,W/REF,horse,0.45,0.6564893019976716,0.021003188091903047
coco,W/SG14,W/REF,motorbike,0.45,0.6235431263967348,0.006021825084586596
coco,W/SG14,W/REF,person,0.45,0.22559080525934666,0.006624164957853146
coco,W/SG14,W/REF,potted plant,0.45,0.6434839707157685,0.0004630257045000752
coco,W/SG14,W/REF,sheep,0.45,0.8750781362481544,0.029518971759139645
coco,W/SG14,W/REF,sofa,0.45,0.5942301307973779,0.0040368700215566156
coco,W/SG14,W/REF,train,0.45,0.6111123366981827,0.0035999537748235833
coco,W/SG14,W/REF,tv,0.45,0.6409098673919524,-0.056009332404466816
