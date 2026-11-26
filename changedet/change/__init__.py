"""Change-detection methods (Yvonne).

Contract every change-detection function must follow
----------------------------------------------------
    def change_map(ref: np.ndarray, aligned: np.ndarray, valid: np.ndarray) -> np.ndarray

* ``ref``: RGB uint8 reference image (H x W x 3).
* ``aligned``: the target already warped onto the reference grid (same shape as
  ``ref``). Pixels outside ``valid`` are zero (no data) and must be ignored.
* ``valid``: bool H x W, the pixels that will be scored.
* Return a float array (H x W). Higher means more likely changed. Only the
  ordering matters: the evaluator sweeps thresholds for precision-recall curves
  and also applies a fixed rule (Otsu or a given value). Use the same scale on
  every image; per-image normalisation to [0, 1] would turn noise into
  "changes" on unchanged pairs.
* No NaN or +inf inside ``valid``.

Run one with ``--change changedet.change.<module>:<function>``. Post-processing
(morphology, graph cuts, ...) belongs inside the function; to evaluate a binary
mask, return it as 0/1 floats.
"""
