"""ECE 549 final project: classical change detection.

Modules
-------
geometry      homography conventions and helpers (read this first)
synth         synthetic benchmark generator   (python -m changedet.synth)
datasets      loaders: synthetic benchmark, HPatches
metrics       registration and change-detection metrics
evaluate      evaluation harness              (python -m changedet.evaluate)
plots         figures and run comparisons     (python -m changedet.plots)
registration  registration methods (contract in registration/__init__.py)
change        change-detection methods (contract in change/__init__.py)
"""
__version__ = "0.1.0"
