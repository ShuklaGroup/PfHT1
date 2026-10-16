# MSM and kinetic analysis

These scripts replace the duplicated apo and glucose-bound workflows with one
set of command-line programs. `featurize_transport_coordinates.py` produces
15 protein contact distances and the TM7b angle, and conditionally adds the
two glucose coordinates;
`build_msm.py` performs tICA, clustering, and maximum-likelihood MSM fitting;
`bootstrap_msm.py` resamples whole trajectories; and `macrostate_mfpt.py`
calculates pairwise MFPTs for gate-defined macrostates.

The feature pickles consumed by `build_msm.py` contain a list of arrays, one
array per continuous trajectory. Frames containing non-finite values must be
removed before fitting while preserving trajectory boundaries.
