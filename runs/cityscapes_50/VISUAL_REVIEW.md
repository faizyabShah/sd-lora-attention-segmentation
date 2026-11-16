# Representative visual review

Exploratory visual observations, not pixel accuracy measurements. Thresholds
were fixed before capture; nothing below changes masks or selects a promoted
method. All comparisons refer to post-R2 masks. The full browser contains all
50 images for every class, including classes absent from an image.

## Observed examples

| Class / image | Threshold behavior and method differences |
|---|---|
| Person, cs_0030 | All methods isolate the nearby pedestrian reasonably at 0.55. At 0.70 standard BRACE trims the head and lower legs; SCLIP-seeded/fusion retain more of the body. |
| Bicycle, cs_0194 | All three methods look similar. 0.40 includes much wall; 0.55 gives a tighter coarse region; 0.70 loses thin bicycle structure. Wheel interiors remain filled rather than segmenting spokes precisely. |
| Car, cs_0485 | SCLIP-seeded and fusion masks are tighter than BRACE at 0.40. At 0.55 all three broadly localize the car; 0.70 trims some boundaries. |
| Sky, cs_0485 | Standard BRACE includes much of the buildings at every tested threshold. SCLIP-seeded BRACE separates the sky far better at 0.55/0.70. Fusion still includes a building at 0.55 and becomes much tighter at 0.70. |
| Building, cs_0485 | At 0.55 SCLIP-seeded/fusion suppress the foreground car and much sky better than BRACE. Higher thresholds further shrink building coverage. This behavior differs from cs_0030, where the SCLIP area filter zeros the building seed. |
| Sidewalk, cs_0030 | BRACE spills onto the pedestrian's legs at 0.40/0.55. SCLIP-seeded and fusion better exclude the person, but 0.70 removes a substantial patch of pavement. |
| Road, cs_0030 | BRACE's 0.55 mask follows much of the paving; SCLIP-seeded spills onto facades and trees even at 0.70. The distinction between road and sidewalk is visually ambiguous in this cobbled scene. |
| Vegetation, cs_0030 | BRACE is class-confused: its 0.70 mask concentrates on the pedestrian rather than the trees. SCLIP's area filter suppresses vegetation, leaving SCLIP-seeded and fusion empty. |
| Absent car, cs_0030 | There is no visible car. BRACE nevertheless returns foreground (including the pedestrian) at all three thresholds. SCLIP-seeded and fusion are empty. This illustrates that per-class min-max masks are not presence detectors. |

Additional completed class representatives:

| Class / image | Threshold behavior and method differences |
|---|---|
| Motorcycle, cs_0083 | All methods produce similar, useful coarse object masks at 0.40/0.55; 0.70 trims some wheel/boundary detail. No obvious fusion advantage. |
| Rider, cs_0450 | None cleanly separates the rider from the bicycle. At 0.55 all include substantial bicycle/ground; at 0.70 SCLIP-seeded loses the upper body. BRACE tightens more, but retains bicycle structure. |
| Sidewalk, cs_0361 | SCLIP-seeded/fusion follow the paths better than BRACE. 0.70 better removes the central grass triangle but also shrinks valid paving. BRACE includes distant buildings and grass at every threshold. |
| Vegetation, cs_0361 | SCLIP-seeded better follows the tree canopy, particularly at 0.55/0.70. BRACE favors the buildings/grass band and misses much canopy. Fusion inherits some of the misplaced grass/building support. |
| Terrain, cs_0361 | No tested threshold fixes this case. BRACE favors a horizon/building band instead of cleanly recovering grass; SCLIP-seeded/fusion are empty because the terrain seed was filtered out. |

Final class representatives:

| Class / image | Threshold behavior and method differences |
|---|---|
| Road, cs_0337 | BRACE's 0.55/0.70 masks follow the asphalt reasonably, with some boundary/grass spill. SCLIP-seeded and fusion are empty at all thresholds: the fixed filter suppresses this visible road class. |
| Bus, cs_0376 | Three methods are nearly identical. 0.40 adds a small background strip; 0.55 is tighter. 0.70 removes more of the front windshield and other bus pixels. |
| Truck, cs_0377 | At 0.40 BRACE includes some facade support; SCLIP-seeded/fusion are slightly tighter. All broadly localize the truck at 0.55. At 0.70 bumper/wheel/cab boundaries lose coverage. |
| Train/tram, cs_0354 | Similar results for all methods. 0.40 includes a wider background strip, 0.55 retains most of the tram, and 0.70 removes roof/window/body pixels. |

All 14 classes have now been inspected in completed post-CRF comparison
sheets. These are fixed class representatives and a few additional failure
cases, not an exhaustive quality grading of all 700 image-class sheets.

## Seed suppression on visually confirmed primary subjects

The benchmark curation establishes presence of these primary subjects; these
are counts of empty seeds, not pixel accuracy measurements. SCLIP's fixed
area filter leaves the seed identically zero in:

| Class | Empty SCLIP seeds / known-presence images |
|---|---:|
| Person | **6 / 12** |
| Bicycle (parked or ridden) | **3 / 14** |
| Rider | **2 / 6** |
| Car | 0 / 6 |
| Bus | 0 / 4 |
| Truck | 0 / 4 |
| Train/tram | 0 / 4 |
| Motorcycle/scooter | 0 / 4 |

Exact image IDs are in `primary_class_seed_diagnostics.json`. For example,
cs_0064 contains a clear pedestrian: BRACE roughly recovers the full body at
0.40/0.55, but both SCLIP-seeded and fusion masks are empty at all thresholds.
Zero seed × any A3 seed stays zero, so a final-threshold change cannot fix
these geometric-fusion failures. The area filter is kept as committed; it
has not been tuned away after observing these results.

## Interpretation

0.55 is a useful reference for visual inspection of some foreground objects;
it is not an estimated optimal threshold. Higher thresholds can reduce spill
but also remove valid thin structures, heads, feet or object boundaries.
SCLIP/fusion improvements and failures depend strongly on class and scene.
An empty SCLIP seed remains empty in geometric fusion, so changing the final
threshold cannot recover it. No universal winning method or threshold is
established by these images without masks.
