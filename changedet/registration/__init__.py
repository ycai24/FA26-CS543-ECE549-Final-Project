"""Registration methods (Yuting: baselines, Shaun: cascade / improvements).

Contract every registration function must follow
------------------------------------------------
    def register(ref: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray | None, dict]

* ``ref``, ``tgt``: RGB uint8 images (H x W x 3). They may differ in size.
* Return ``H`` (3x3 float) mapping TARGET pixel coords to REFERENCE pixel coords,
  i.e. ``cv2.findHomography(tgt_pts, ref_pts, ...)``. Return ``None`` if
  registration failed; it is scored as a failure rather than crashing the run.
* Return an ``info`` dict of scalars (method name, number of matches, inliers,
  which cascade branch fired, ...). The evaluator copies it into the per-pair CSV
  with a ``reg_`` prefix.

Run one with the evaluator using ``--register changedet.registration.<module>:<function>``.
"""
