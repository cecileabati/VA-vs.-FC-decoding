# VA vs FC decoding of affective states

Analysis code for a preregistered study comparing two fMRI feature
representations **voxel-wise activation (VA)** and **functional connectivity
(FC)** for decoding affective states from an emotional face-matching task.

- **Sample:** Münster Neuroimaging Cohort, 389 healthy participants scanned;
  n = 372 after exclusions (see [Key analysis choices](#key-analysis-choices))
- **Task:** EFMT, mixed block/event design, 210 volumes, TR = 2.1 s (438s s run)
- **Conditions:** 5: shape blocks (36 s) plus neutral / angry / happy / fearful
  face events (2 s, 12x)
- **Parcellation:** Brainnetome (246 regions), plus a 30-ROI subset (based on Fusar-Poli et al., 2009)
- **Preregistration:** *https://osf.io/gub38/overview?view_only=77bf180f6e784c1493d0905a083ab102*

---

## Setup

Every absolute path is read from a single file that is **not** in this
repository, so no machine-specific path appears in any script.

```bash
cp paths.example.json paths.json     # then edit the values
python3 project_paths.py             # prints every path and whether it exists
```

`project_paths.py` exits non-zero if an input path is missing.

| key | points at |
|---|---|
| `vault_root` | project root holding `sample/`, the Brainnetome atlas, SPM and CONN |
| `conn_base` | the CONN preprocessing folder (absolute, or relative to `vault_root`) |
| `perm_scratch` | node-local scratch for per-permutation PHOTONAI logging |

Everything else — beta-map folders, CONN first-level folders, the subject
mapping CSV, the ROI mask, is a fixed layout beneath those and is derived by
the loaders, so this repository documents the directory structure it expects
without containing anyone's paths.

The file is searched for in this order, so it can live outside the working tree
entirely, which is the recommended setup: a file git cannot see cannot be
committed by accident.

1. `$VAFC_PATHS`
2. `paths.json` next to the loader
3. `config/paths.json`
4. `~/.vafc_paths.json`

Both read the same file. `project_paths.py` exposes module constants,
`project_paths.m` returns a struct.

---

## Pipeline

Run in this order. Data is not included, see [Data](#data).

| # | Script | Does | Output |
|---|---|---|---|
| 1 | `setup/subject_ID_mapping.m` | Reads CONN's project file, builds the CONN-index → original-ID lookup | `conn_subject_mapping.csv` |
| 2 | `setup/mask_30roi.m` | Binary mask from 30 Brainnetome regions | `roi30_mask.nii` |
| 3 | `preprocessing/final_denoising_script.m` | CONN denoising batch, 32 confound regressors | denoised ROI timeseries |
| 4 | `firstlevel/combined_wb_roi_beta_maps_final.m` | SPM first level, 5 conditions + 32 nuisance regressors (set `SCOPE`) | beta maps per condition |
| 5 | `decoding/decoding_pipeline_final.py` | PHOTONAI nested CV + permutation tests | accuracies, p-values, null distributions |

Step 3 does not run start to finish on its own: it sets the denoising
structure, pauses for an inspection of `CONN_x.Preproc.confounds`, runs it, and
then needs one manual step in the CONN GUI, noted at the end of that file.

**FC matrices** come from CONN's first-level ROI-to-ROI analysis (weighted GLM,
`hrf` weighting), configured in the CONN GUI rather than scripted. Two analyses
are defined: `Brainnetome` (246 regions) and `Brainnetome_30ROIs`.

### Scope switch

Two feature spaces are analysed. Both the first-level GLM and the decoding
script have a `SCOPE` variable near the top that must be set to the same value,
and each is run twice, once per scope.

**`firstlevel/combined_wb_roi_beta_maps_final.m`**

```matlab
SCOPE = 'whole_brain';   % mask = full 246-region Brainnetome atlas
                         % output -> beta_maps/Beta_maps_brainnetome_32conf/
SCOPE = 'roi';           % mask = roi30_mask.nii
                         % output -> beta_maps/Beta_maps_brainnetome_30ROI_32conf/
```

**`decoding/decoding_pipeline_final.py`** (in the configuration block)

```python
SCOPE = "whole_brain"    # reads Brainnetome/ + Beta_maps_brainnetome_32conf/
SCOPE = "roi"            # reads Brainnetome_30ROIs/ + the 30ROI beta maps
```

| Scope | VA features | FC features |
|---|---|---|
| `whole_brain` | ~127,000 voxels | 30,135 edges (246 regions) |
| `roi` | subset inside the 30-ROI mask | 435 edges (30 regions) |

The two must match: running the GLM as `whole_brain` and the decoding as `roi`
would read beta maps and FC matrices from different feature spaces. Nothing
enforces this, so check it before each run.

The decoding script also has an `ANALYSIS` variable selecting the hypothesis
(`H1_faces_vs_shapes`, `H2_emotional_vs_neutral`, `H3_emotion_categories`). It
is set once per run and, with `SCOPE`, determines every output filename, so
results from different runs cannot overwrite one another.

---

## Analysis specifics 

**Denoising (both arms share the same nuisance model).**
5 WM + 5 CSF aCompCor components and 6 motion parameters, all with first
derivatives = **32 regressors**. The WM/CSF components are taken from CONN's
Setup-extracted ROI timeseries (`data/ROI_Subject###_Session001.mat`,
pre-denoising) and passed to SPM via `multi_reg`, so both arms regress out
literally the same timeseries.

- FC arm: bandpass 0.01–0.1 Hz, no detrending, despiking after regression
  (RegBP). df = (210 − 1 − 32) × 0.378 = 66.9
- VA arm: high-pass 128 s, AR(1) prewhitening, no bandpass. Residual df ≈ 164

**Masking.** `mthresh = -Inf` disables SPM's implicit masking; the explicit mask
is the atlas (or the 30-ROI mask). Both arms therefore have identical spatial
space, so the VA/FC comparison contrasts representations rather than different brain areas included.

**Decoding.** Nested cross-validation with `StratifiedGroupKFold`, grouped by
subject. Permutations shuffle condition labels **within** subject; FC and beta
share permutation indices so the FC-minus-beta difference test is paired.

**Exclusions.** Two participants excluded by hand (one with partial brain
coverage, one follow-up session), all remaining follow-up sessions, and the
entire adolescent subsample (IDs beginning `JN`). Only baseline scans of the
adult sample enter the analysis, leaving **n = 372**. The `noJN` tag in output
filenames marks the adolescent exclusion.

---

## Data

Not in this repository. The data set is not publicly available, but these
scripts can be pointed at another data set by editing `paths.json`.

---

## Dependencies

- MATLAB R2024a, SPM12, CONN 22.v2407
- Python 3.11: numpy, scipy, nibabel, joblib, scikit-learn
- [PHOTONAI](https://github.com/wwu-mmll/photonai) nested CV framework
  (Leenings et al.)

---

## Citation

*(paper reference, once published)*
