# SG18 middle versus late call diagnostic

Late 21–30 is a diagnostic-only map, not a selectable method. All values use the original SG18 cached maps.

## VOC at fixed threshold 0.425

| Window | mIoU | Precision | Recall | CA-relative TP add | FP add | distant FP >16 | separate FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| middle | 0.7959 | 0.8646 | 0.9016 | 6,073,586 | 2,053,175 | 961,952 | 78,319 |
| late | 0.7890 | 0.8578 | 0.8997 | 5,863,778 | 2,141,121 | 975,584 | 83,104 |
| middle_plus_late | 0.7936 | 0.8616 | 0.9017 | 5,994,112 | 2,064,454 | 948,801 | 90,397 |

Middle → middle+late threshold crossings: TP gained 261,001, TP lost 255,053, FP gained 399,322, FP removed 226,040. Distant FP gained/removed (>16 px): 155,712/101,371.

## VOC at fixed threshold 0.45

| Window | mIoU | Precision | Recall | CA-relative TP add | FP add | distant FP >16 | separate FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| middle | 0.7941 | 0.8772 | 0.8819 | 5,370,529 | 1,539,078 | 699,620 | 53,984 |
| late | 0.7868 | 0.8713 | 0.8784 | 5,047,502 | 1,563,197 | 681,784 | 65,386 |
| middle_plus_late | 0.7916 | 0.8745 | 0.8814 | 5,233,753 | 1,531,674 | 683,050 | 55,293 |

Middle → middle+late threshold crossings: TP gained 291,931, TP lost 311,326, FP gained 352,774, FP removed 208,377. Distant FP gained/removed (>16 px): 120,505/84,849.

## COCO at fixed threshold 0.425

| Window | mIoU | Precision | Recall | CA-relative TP add | FP add | distant FP >16 | separate FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| middle | 0.6285 | 0.6755 | 0.8670 | 8,479,746 | 9,834,705 | 7,593,273 | 1,122,681 |
| late | 0.6191 | 0.6789 | 0.8451 | 7,543,957 | 8,975,961 | 6,921,307 | 1,218,335 |
| middle_plus_late | 0.6240 | 0.6763 | 0.8576 | 8,047,520 | 9,464,975 | 7,309,527 | 1,103,063 |

Middle → middle+late threshold crossings: TP gained 178,332, TP lost 581,249, FP gained 674,271, FP removed 929,652. Distant FP gained/removed (>16 px): 445,262/656,999.

## COCO at fixed threshold 0.45

| Window | mIoU | Precision | Recall | CA-relative TP add | FP add | distant FP >16 | separate FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| middle | 0.6343 | 0.6956 | 0.8470 | 7,729,556 | 8,217,193 | 6,372,719 | 905,372 |
| late | 0.6251 | 0.7011 | 0.8233 | 6,742,887 | 7,235,963 | 5,601,340 | 1,037,371 |
| middle_plus_late | 0.6307 | 0.6978 | 0.8377 | 7,302,756 | 7,771,380 | 6,028,903 | 942,716 |

Middle → middle+late threshold crossings: TP gained 185,089, TP lost 580,629, FP gained 587,664, FP removed 919,067. Distant FP gained/removed (>16 px): 371,860/643,100.
