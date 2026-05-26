# Legacy Source

This directory contains the previous fixed canonical 8-phase controller stack.

It is kept as a reference baseline, but it is no longer the primary
implementation path for the project. New work should target the movement-aware
controller under `src/movement/`.

The legacy code may still contain imports that reference its old location
under `src.environment`, `src.model`, `src.training`, or `src.utils`. Those
imports are intentionally not repaired as part of the initial movement-policy
checkpoint.
