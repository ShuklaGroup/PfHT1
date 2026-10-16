# Code provenance

This folder was assembled on 2026-10-15 from the analysis code used for the
PfHT1 manuscript.

- `binding_pocket_tm7b/` contains the retained apo-reference pocket analysis.
  The calculation code is unchanged apart from resolving the project and
  analysis roots through `project_paths.py`, so the archived copy does not
  depend on the location of this code folder.
- `msm_kinetics/` consolidates the equivalent apo and glucose-bound tICA,
  clustering, MSM, bootstrap, and MFPT scripts into condition-independent
  command-line programs. `featurize_transport_coordinates.py` likewise
  consolidates the two transport-coordinate featurizers and retains their
  condition-specific output column order.
- `residue_contacts/residue_contact_scores.py` consolidates the original RRCS
  matrix and difference scripts. It preserves the 3.23/4.63 A scoring rule and
  the side-chain-only rule for residues separated by four or fewer positions.
- `yeast_growth/plot_pooled_normalized_growth_rates.py` is the final pooled
  growth-analysis script with input and output paths exposed as arguments.
- `free_energy_landscapes/weighted_free_energy.py` is the common form of the
  repeated condition-specific weighted free-energy plotting workflow.

The public PfHT1 repository already contains the GAT, VAE, transition-path,
and graph-preprocessing code. Those files were deliberately excluded here.
