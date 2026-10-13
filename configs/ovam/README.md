# OVAM optimized VOC tokens

`tokens_VOC.npy` is copied unchanged from the OVAM repository
(<https://github.com/vpulab/ovam>, `docs/assets/tokens_VOC.npy`, commit
`12afd87cacb2c154681da58635b75cf9451db406`). It is a pickled dict with one
768-dimensional text-encoder-output vector per VOC class, keyed by class name
(`aeroplane` … `tv`; multi-word classes use a hyphen, e.g. `dining-table`).

Per the OVAM paper (Sec. 4.1, Supp. A.1), each token was optimized on one
hand-annotated 512×512 image per class, generated with "A photograph of a
⟨classname⟩" and seeds that are not in the evaluation sets.

OVAM is released under the MIT license; its notice is in `LICENSE_OVAM`.
