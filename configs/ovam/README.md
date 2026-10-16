# OVAM optimized VOC tokens

`tokens_VOC.npy` is copied unchanged from the OVAM repository
(<https://github.com/vpulab/ovam>, `docs/assets/tokens_VOC.npy`, commit
`12afd87cacb2c154681da58635b75cf9451db406`). It is a pickled dict with one
768-dimensional text-encoder-output vector per VOC class, keyed by class name
(`aeroplane` … `tv`; multi-word classes use a hyphen, e.g. `dining-table`).

OVAM's public archive (`evaluation/ovam/voc_sim_train/`) holds the training
set: 63 images generated with "A photograph of a ⟨classname⟩" (3 per class; 2
for `cat`, `horse` and `tv`), 59 hand-drawn masks, and 57 per-image tokens.
None of its seeds is a VOC-Sim evaluation seed. **Each released token equals
the mean of that class's 2–3 per-image tokens** (row 1 of each `(2, 768)`
per-image file; maximum difference below 10⁻⁶), so the released tokens are
2–3-shot. Earlier versions of this note said one image per class, from a
summary of the OVAM paper (Sec. 4.1, Supp. A.1); the paper's wording is to be
checked.

OVAM is released under the MIT license; its notice is in `LICENSE_OVAM`.
