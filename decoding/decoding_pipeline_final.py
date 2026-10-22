"""
Hierarchical decoding analysis (H1/H2/H3) -> H4 comparison of FC vs beta.

Set SCOPE (roi vs. whole brain) and ANALYSIS below; every output file is named
from that combination. Defaults to 1000 permutations per feature type across 40
parallel workers, both overridable with --n_perms and --n_jobs. Subject IDs come
from a mapping CSV, because the SPM subject numbers differ from the CONN ones.


WHY WITHIN-SUBJECT SHUFFLING
----------------------------
Each subject contributes one row per condition, so condition is a WITHIN-subject
factor and rows from one subject are not independent. We therefore permute the
condition labels ONLY among a subject's own rows (exchangeability block =
subject). This preserves each subject's class composition and the subject-level
dependence and is consistent with the StratifiedGroupKFold splitting.
"""

import os  # file paths, env, vars, directory creation

# Limit BLAS threads per process: each permutation worker is single-threaded so
# that N parallel jobs map cleanly onto N cores without oversubscription.
os.environ["OMP_NUM_THREADS"] = "1" # OpenMP threads -> 1
os.environ["MKL_NUM_THREADS"] = "1" # Intel MKL threads -> 1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS threads -> 1


import re  # regex -> parse subject IDs, match follow-up tags
import csv # read subject mapping CSV (we have different IDs from SPM and CONN)
import shutil    # remove each worker's throwaway project_folder
import tempfile  # unique per-worker project_folder -> kills the logging-file race
import argparse # parse --n_perms / --n_jobs from the command line
import traceback # full tracebacks for durable per-perm failure logs
import numpy as np # array, RNG, permutation, p-value
import scipy.io as sio # loads .mat files (FC matrices)
import nibabel as nib # load nifti files (beta maps)
import joblib                        # pickle the fitted pipelines for later interpretation
from joblib import Parallel, delayed # runs permutation tests across workers

from collections import Counter # prints class frequency

# Project-local: every absolute path, resolved from the gitignored paths.json
# via project_paths.py. Keeps server paths out of version control.
from project_paths import (MAPPING_CSV, PROJECT_FOLDER, FAIL_DIR, CKPT_DIR,
                           PERM_SCRATCH as PERM_PROJECT_FOLDER,
                           for_scope, ensure_output_dirs)


# 1. CONFIGURATION (module level)
# The four folders below are imported above, not hardcoded:
#
#   PROJECT_FOLDER       results are stored here (shared vault)
#   PERM_PROJECT_FOLDER  per-perm PHOTONAI logging -> fast, reliable node-local
#                        disk, never the shared vault. save_output=False writes
#                        nothing scientific; this is only the PARENT dir under
#                        which each worker creates its OWN unique subfolder (see
#                        one_perm) so their setup-error logs never collide.
#                        Still overridable with $PERM_PROJECT_FOLDER.
#   FAIL_DIR             durable per-perm failure logs, written to the VAULT
#                        project folder (co-located with the results CSV, per
#                        Cecile). One file per failed perm so parallel workers
#                        never contend on a shared handle. Written rarely (only
#                        on failure) -> no race concern.
#   CKPT_DIR             checkpoints, because of failed permutation runs, so we
#                        can resume from the last checkpoint
ensure_output_dirs()

CONDITION_BETA_INDEX = {
    "Condition001": 1, "Condition002": 2, "Condition003": 3,
    "Condition004": 4, "Condition005": 5,
}

# scope: whole-brain vs. 30-ROI
SCOPE = "whole_brain"   # "roi" or "whole brain" -> change dempeding on analysis
SCOPE_TAG = "_" + SCOPE    # adds "_whole_brain" or "_roi"
CONF_TAG = "_32conf" # adjusted denoising (32 confounds) 
PERM_TAG = "_1000perm_noJN"  

FC_FOLDER, BETA_FOLDER = for_scope(SCOPE)   # raises ValueError on an unknown SCOPE

# Choose for ech run the according hypothesis
ANALYSIS_CONFIGS = {
    "H1_faces_vs_shapes": {                                                  # H1: any face vs shapes
        "positive_classes": ["Condition002", "Condition003", "Condition004", "Condition005"],  # all faces -> label 1
        "negative_classes": ["Condition001"],                               # shapes -> label 0
        "pipe_name": "h1_faces_vs_shapes",                                  # tag used in filenames
    },
    "H2_emotional_vs_neutral": {                                            # H2: emotional faces vs neutral faces
        "positive_classes": ["Condition003", "Condition004", "Condition005"],  # emotional -> 1
        "negative_classes": ["Condition002"],                               # neutral -> 0
        "pipe_name": "h2_emotional_vs_neutral",
    },
    "H3_emotion_categories": {                                              # H3: multiclass among emotions
        "classes": {"Condition003": 0, "Condition004": 1, "Condition005": 2},  # explicit class -> integer label
        "pipe_name": "h3_emotion_categories",
    },
}
ANALYSIS = "H2_emotional_vs_neutral" # change according to your analysis
cfg = dict(ANALYSIS_CONFIGS[ANALYSIS])      # COPY so we don't mutate the shared config dict
cfg["pipe_name"] += SCOPE_TAG + CONF_TAG + PERM_TAG   # append the scope tag to the output name

N_CONFIGS = 40                              # random-search configurations
PERM_MASTER_SEED = 20240601                 # fixed -> reproducible + shared fc/beta shuffles
METRIC = "balanced_accuracy"                # scoring metric everywhere
N_PERMS_DEFAULT = 1000                      # default permutations per feature type
N_JOBS_DEFAULT = 40                         # parallel workers

FEATURE_PARAMS = {
    "fc":   {"pca": (30, 200), "C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "beta": {"pca": (50, 400), "C": [0.001, 0.01, 0.1, 1, 10, 100]},
}

EXCLUDE_SUBJECTS = {"N439_prae", "N198_10J"} # half brain and follow up appointment
FOLLOWUP_RE = re.compile(r"_\d+J$") # follow-up appointment in subject ID (e.g., N198_10J)
ADOLESCENT_RE = re.compile(r"^JN")  # JN* = adolescent subsample (<18), excluded

# 2. HELPER FUNCTIONS

def load_subject_map(csv_path):
    """Read the mapping CSV -> {conn_subject_number: original_id}."""
    mapping = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mapping[int(row["conn_subject"])] = row["original_id"].strip()
    return mapping


def keep_subject(sid):
    """Inclusion filter: False for hand-excluded subjects, follow-up sessions,
    and the adolescent (JN) subsample."""
    if sid in EXCLUDE_SUBJECTS:
        return False
    if FOLLOWUP_RE.search(sid):
        return False
    if ADOLESCENT_RE.match(sid):
        return False
    return True


def flatten_upper(matrix, k=1):
    """Vectorise upper triangle (k=1) of a matrix -> 1-D vector"""
    return matrix[np.triu_indices_from(matrix, k=k)]


def resolve_beta_path(subject_dir, beta_index):
    """Return a subject's beta file for a given index, trying .nii then .nii.gz."""
    for ext in (".nii", ".nii.gz"):
        p = os.path.join(subject_dir, f"beta_{beta_index:04d}{ext}")  # e.g. beta_0003.nii
        if os.path.exists(p):
            return p
    return None


def load_volume(path):
    """Load a NIfTI volume as a float32 numpy array (data only)."""
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)

# 3. DATA LOADERS  (each keyed by canonical original_id)

def load_fc(conditions, conn2id):
    """Load FC features"""
    out = {}
    for fname in sorted(os.listdir(FC_FOLDER)):
        if not fname.endswith(".mat"):
            continue
        base = fname.replace(".mat", "")
        subj_str, cond = base.rsplit("_", 1)
        if cond not in conditions:
            continue
        m = re.search(r"Subject(\d+)", subj_str)
        if not m:
            continue
        conn_num = int(m.group(1))
        sid = conn2id.get(conn_num)
        if sid is None:
            print(f"  WARNING: conn_subject {conn_num} not in mapping -> skipped")
            continue
        if not keep_subject(sid):
            continue
        mat = sio.loadmat(os.path.join(FC_FOLDER, fname))
        if "Z" not in mat:
            continue
        z = mat["Z"]
        if z.ndim == 3:
            z = np.squeeze(z)
        if z.ndim != 2:
            continue
        out.setdefault(sid, {})[cond] = flatten_upper(z)
    return out


def load_beta_paths(conditions):
    """Locate beta-map PATHS (not data) -> {original_id: {condition: filepath}}"""
    out = {}
    for entry in sorted(os.listdir(BETA_FOLDER)):
        sdir = os.path.join(BETA_FOLDER, entry)
        if not os.path.isdir(sdir):
            continue
        if not re.search(r"\d", entry):
            continue
        sid = entry
        if not keep_subject(sid):
            continue
        for cond in conditions:
            p = resolve_beta_path(sdir, CONDITION_BETA_INDEX[cond])
            if p is not None:
                out.setdefault(sid, {})[cond] = p
    return out


# 4. PIPELINE SETUP  (parameterised by feature type)

def build_pipeline(feature_type, name_suffix="", save_output=True, project_folder=None):
    """Construct the PHOTONAI Hyperpipe for one feature type (fc | beta).

    project_folder: if given, overrides where this pipe writes its logs/output.
    Perm workers pass a UNIQUE tempdir here so their photonai_setup_errors.log
    never collides (root cause of the old n<100 attrition)."""
    from photonai.base import Hyperpipe, PipelineElement, Switch, OutputSettings
    from photonai.optimization import IntegerRange
    from sklearn.model_selection import StratifiedGroupKFold

    params = FEATURE_PARAMS[feature_type] # PCA range & C grid for feature type
    name = f"{feature_type}_{cfg['pipe_name']}{name_suffix}" # unique pipe name

    if project_folder is None:                                # default behaviour unchanged
        project_folder = PROJECT_FOLDER if save_output else PERM_PROJECT_FOLDER

    my_pipe = Hyperpipe(
        name, # pipe name
        optimizer='random_grid_search',
        optimizer_params={'n_configurations': N_CONFIGS}, random_seed=42,
        metrics=['balanced_accuracy', 'accuracy'],
        best_config_metric='balanced_accuracy',
        outer_cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
        inner_cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
        calculate_metrics_across_folds=True,
        use_test_set=True, # score the best config on the held-out outer folds
        verbosity=0, # quiet output
        project_folder=project_folder,
        output_settings=OutputSettings(save_output=save_output),
    )

    my_pipe += Switch(
        'StandardizationSwitch',
        [PipelineElement('StandardScaler'),
         PipelineElement('MinMaxScaler')],
    )

    my_pipe += PipelineElement(
        'PCA', # dimensionality reduction
        hyperparameters={'n_components': IntegerRange(*params["pca"])},
        test_disabled=True, # also try no PCA
    )

    svm = PipelineElement(
        'SVC',
        hyperparameters={'kernel': ['linear', 'rbf'],
                         'C':      params["C"],
                         'gamma':  ['auto', 'scale']},
        class_weight='balanced',
        max_iter=int(1e6),
    )

    tree = PipelineElement(
        'DecisionTreeClassifier',
        hyperparameters={'max_depth':         IntegerRange(3, 15),
                         'min_samples_split': IntegerRange(2, 5),
                         'min_samples_leaf':  IntegerRange(1, 5),
                         'criterion':         ['gini', 'entropy']},
        class_weight='balanced',random_state=42)

    my_pipe += Switch('EstimatorSwitch', [svm, tree])

    return my_pipe

def mean_outer_test_ba(pipe):
    """Mean held-out OUTER-CV balanced accuracy of the selected model.
    Reads only outer-test scores (never inner-CV); the metrics_copied_from_inner
    guard refuses an inner-CV-derived score (would only arise if use_test_set=False)."""
    outer = pipe.results.outer_folds                          # per-outer-fold result objects
    for of in outer:                                          # guard: never report inner-CV numbers
        val = of.best_config.best_config_score.validation     # held-out outer-test score of the fold's winner
        if getattr(val, "metrics_copied_from_inner", False):  # True only if use_test_set=False
            raise RuntimeError("Outer-test metrics copied from inner CV (use_test_set=False?).")
    try:
        v = pipe.results.get_test_metric(METRIC, operation="mean")   # PHOTONAI's mean outer-test metric
        if v is not None:
            return float(v)
    except Exception:
        pass                                                  # fall through to the manual mean
    return float(np.mean([of.best_config.best_config_score.validation.metrics[METRIC]   # per-fold outer-test BA...
                          for of in outer]))                                             # ...averaged (same quantity)


def permuted_labels(y, groups, perm_index):
    """Within-subject shuffle. Seeded only by (PERM_MASTER_SEED, perm_index), so
    fc-perm-i and beta-perm-i get identical labels -> valid PAIRED difference."""
    rng = np.random.default_rng([PERM_MASTER_SEED, perm_index])
    y_perm = y.copy()                                            # start from the true labels
    for g in np.unique(groups):                                  # for each subject block
        idx = np.flatnonzero(groups == g)                        # that subject's row indices
        y_perm[idx] = rng.permutation(y[idx])                    # shuffle labels WITHIN subject
    return y_perm                                                # permuted label vector



# 5. DATA PREPARATION

def prepare_data():
    """Build aligned FC + beta feature matrices, labels, and subject groups"""
    if "classes" in cfg:
        COND_NUM = dict(cfg["classes"])
    else:
        COND_NUM = {c: 0 for c in cfg["negative_classes"]}
        COND_NUM.update({c: 1 for c in cfg["positive_classes"]})
    CONDITIONS = list(COND_NUM.keys())

    print(f"Running analysis: {ANALYSIS}")
    print(f"Conditions used : {CONDITIONS}")
    print(f"Label mapping   : {COND_NUM}")

    conn2id = load_subject_map(MAPPING_CSV)
    if conn2id.get(309) != "N439_prae":
        print(f"  WARNING: conn_subject 309 maps to {conn2id.get(309)!r}, not 'N439_prae'.")

    fc_data    = load_fc(CONDITIONS, conn2id)
    beta_paths = load_beta_paths(CONDITIONS)

    def complete(d): # subjects that are both complete n FC & beta for ll conditions
        return {s for s, cd in d.items() if all(c in cd for c in CONDITIONS)}

    common = sorted(complete(fc_data) & complete(beta_paths))
    print(f"FC subjects: {len(complete(fc_data))} | "
          f"beta subjects: {len(complete(beta_paths))} | COMMON (used): {len(common)}")
    if not common:
        raise RuntimeError("No overlapping subjects - check MAPPING_CSV / folder paths.")

    pairs  = [(s, c) for s in common for c in CONDITIONS] # cannonical row order -> subject x condition
    y      = np.array([COND_NUM[c] for (s, c) in pairs]) # label per row
    groups = np.array([s           for (s, c) in pairs]) # subject ID per row

    X_fc = np.vstack([fc_data[s][c] for (s, c) in pairs]).astype(np.float32) # stcked FC matrices

    beta_files = [beta_paths[s][c] for (s, c) in pairs]
    ref_img    = nib.load(beta_files[0])
    ref_shape  = ref_img.shape
    ref_affine = ref_img.affine
    finite_mask = np.ones(ref_shape, dtype=bool)
    nonzero_any = np.zeros(ref_shape, dtype=bool)
    for p in beta_files:
        img = nib.load(p)
        if img.shape != ref_shape:
            raise ValueError(f"Shape mismatch: {p} {img.shape} vs reference {ref_shape}")
        if not np.allclose(img.affine, ref_affine, atol=1e-3):
            print(f"  WARNING: affine mismatch for {p}")
        vol = np.asarray(img.dataobj, dtype=np.float32)
        finite_mask &= np.isfinite(vol)
        nonzero_any |= (np.nan_to_num(vol) != 0.0)
    group_mask = finite_mask & nonzero_any
    print(f"Beta group mask: {int(group_mask.sum())} voxels")

    X_beta = np.vstack([load_volume(p)[group_mask] for p in beta_files]).astype(np.float32)
    nonconst = X_beta.std(axis=0) > 0
    if (~nonconst).any():
        print(f"Dropping {int((~nonconst).sum())} zero-variance voxels")
        X_beta = X_beta[:, nonconst]

    final_flat = np.zeros(group_mask.size, dtype=bool)
    final_flat[np.flatnonzero(group_mask.ravel())[nonconst]] = True
    final_mask = final_flat.reshape(ref_shape)
    mask_path = os.path.join(PROJECT_FOLDER, f"beta_{cfg['pipe_name']}_feature_mask.nii")
    nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), ref_affine), mask_path)
    print(f"Saved beta feature mask -> {mask_path}")

    print(f"X_fc:   {X_fc.shape}")
    print(f"X_beta: {X_beta.shape}")
    print("Class distribution:", Counter(y))
    print("Unique subjects (groups):", len(np.unique(groups)))

    return {"fc": X_fc, "beta": X_beta}, y, groups


# 6. RUN

def one_perm(ft, X, y, groups, i):
    """One permutation (a joblib task): shuffle labels within subject, refit the
    pipeline, return the null balanced accuracy (NaN on failure).

    Two details below are deliberate, and both caused silently dropped
    permutations before they were put in:

    1. Each worker gets its OWN unique project_folder (workdir, removed in
       finally). Every Hyperpipe writes photonai_setup_errors.log into its
       project_folder even with save_output=False, so a shared path makes
       concurrent workers race on that one file -> FileNotFoundError -> NaN.
       Node-local disk alone does not help; it is the PATH that is shared.
    2. The checkpoint write sits in its own try, so a momentarily unwritable
       share cannot discard a score that has already been computed."""
    y_perm = permuted_labels(y, groups, i)                    # within-subject shuffle (seed=(master, i))
    workdir = tempfile.mkdtemp(prefix=f"perm_{ft}_{i}_", dir=PERM_PROJECT_FOLDER)  # unique, node-local
    try:
        pipe = build_pipeline(ft, name_suffix=f"_perm{i}",    # build INSIDE try (construction failure -> NaN, never fatal)
                              save_output=False, project_folder=workdir)
        pipe.fit(np.asarray(X), y_perm, groups=groups)        # fit full nested CV pipeline on permuted labels
        score = mean_outer_test_ba(pipe)                      # null score for this permutation
        # Checkpoint in its OWN try: if the share is briefly unwritable (expired
        # Kerberos ticket, vault hiccup) the write fails, but the score has already
        # been computed and must still reach the parent. Previously this np.save sat
        # in the main try, so a write failure fell through to "return np.nan" and
        # silently discarded hours of finished compute.
        try:
            np.save(os.path.join(CKPT_DIR, f"{ft}_{cfg['pipe_name']}_{i:05d}.npy"),
                    np.array([score]))                        # checkpoint immediately
        except Exception as e:
            print(f"  [{ft}] perm {i}: score {score:.4f} computed but checkpoint "
                  f"FAILED ({type(e).__name__}: {e}) - value kept, recoverable "
                  f"from this log", flush=True)
        return score
    except Exception as e:                                    # construction OR fit failure shouldn't kill the run
        print(f"  [{ft}] perm {i} FAILED: {type(e).__name__}: {e}")   # note it (may age out of tmux -> also file)
        try:
            with open(os.path.join(FAIL_DIR, f"{ft}_{cfg['pipe_name']}_perm{i}.txt"), "w") as fh:
                fh.write(f"[{ft}] perm {i} FAILED: {type(e).__name__}: {e}\n\n" + traceback.format_exc())
        except Exception:
            pass                                              
        return np.nan                                        
    finally:
        shutil.rmtree(workdir, ignore_errors=True)            # clean up the throwaway project_folder


def run(n_perms, n_jobs):
    """Fit true labels, run the parallel permutation nulls, compute + write p-values"""
    features, y, groups = prepare_data()                      # build X_fc, X_beta, y, groups (once)

        # 1) true-label fit per feature type (cached: skipped if already on disk)
    true = {}                                                 # ft: true outer-test BA
    for ft in ("fc", "beta"):                                 # fit FC and beta on the REAL labels
        tfile = os.path.join(PROJECT_FOLDER, f"true_{ft}_{cfg['pipe_name']}.npy")
        if os.path.exists(tfile):                             # resume: reuse the earlier true fit
            true[ft] = float(np.load(tfile)[0])
            print(f"[{ft}] loaded true {METRIC} = {true[ft]:.4f} from disk")
            continue
        print(f"\n=== fitting {ft.upper()} on TRUE labels ===")
        pipe = build_pipeline(ft)                             # save_output=True (writes a results folder)
        pipe.fit(features[ft], y, groups=groups)              # full nested CV fit
        true[ft] = mean_outer_test_ba(pipe)                   # observed score
        np.save(tfile, np.array([true[ft]]))                  # save NOW so a crash can't lose it
        try:                                                  # keep the fitted pipe for later interpretation
            joblib.dump(pipe, os.path.join(PROJECT_FOLDER,
                        f"pipe_{ft}_{cfg['pipe_name']}.joblib"))
        except Exception as e:                                # pickling must never kill the run
            print(f"  [{ft}] could not pickle pipeline: {e}")
        print(f"[{ft}] true {METRIC} = {true[ft]:.4f}")       # log it


     # 2) null distribution: skip a feature type that is already complete,
    #    and within a feature type run only the permutations not yet checkpointed
    null = {}
    for ft in ("fc", "beta"):
        final = os.path.join(PROJECT_FOLDER, f"null_scores_{ft}_{cfg['pipe_name']}.npy")
        if os.path.exists(final):
            null[ft] = np.load(final)
            print(f"[{ft}] loaded {null[ft].size} completed permutations from disk")
            continue

        done = {}
        for i in range(n_perms):
            f = os.path.join(CKPT_DIR, f"{ft}_{cfg['pipe_name']}_{i:05d}.npy")
            if os.path.exists(f):
                done[i] = float(np.load(f)[0])
        todo = [i for i in range(n_perms) if i not in done]
        print(f"\n=== {ft.upper()}: {len(done)} already done, "
              f"{len(todo)} to run on {n_jobs} workers ===")

        if todo:
            scores = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(one_perm)(ft, features[ft], y, groups, i) for i in todo)
            done.update(dict(zip(todo, scores)))

        null[ft] = np.array([done.get(i, np.nan) for i in range(n_perms)])
        np.save(final, null[ft])

    # INSTRUMENTATION: save raw null scores (NaN preserved) + true scores next to the CSV.
    # Lets us see exactly which perm indices (if any) are NaN and plot the null histogram.
    for ft in ("fc", "beta"):
        np.save(os.path.join(PROJECT_FOLDER, f"null_scores_{ft}_{cfg['pipe_name']}.npy"), null[ft])
    np.save(os.path.join(PROJECT_FOLDER, f"true_scores_{cfg['pipe_name']}.npy"),
            np.array([true["fc"], true["beta"]]))
    for ft in ("fc", "beta"):                                 # quick attrition report to stdout/log
        nbad = int(np.sum(~np.isfinite(null[ft])))
        print(f"[{ft}] permutations returned: {null[ft].size} | NaN/failed: {nbad}")

    # 3) p-values
    out_csv = os.path.join(PROJECT_FOLDER, f"permutation_results_{cfg['pipe_name']}.csv")   # output path
    with open(out_csv, "w", newline="") as f:                # open the CSV for writing
        w = csv.writer(f)
        w.writerow(["test", "true_score", "n", "null_mean", "null_sd", "p_value", "min_possible_p"])   # header

        # (1) per-feature above-chance
        print("\n----- results -----")
        for ft in ("fc", "beta"):                            # one above-chance test per feature type
            s = null[ft][np.isfinite(null[ft])]              # drop failed (NaN) permutations
            n = s.size                                       # number of valid permutations
            p = (np.sum(s >= true[ft]) + 1) / (n + 1)        # right-tail p
            print(f"[{ft}] true={true[ft]:.4f} | null {s.mean():.4f}±{s.std():.4f} | n={n} | p={p:.4f}")   # log
            w.writerow([f"{ft}_above_chance", f"{true[ft]:.4f}", n,          # write the row
                        f"{s.mean():.4f}", f"{s.std():.4f}", f"{p:.4f}", f"{1/(n+1):.4f}"])

        # (2) paired FC-vs-beta difference (pair by index -> shared shuffle)  == H4
        both = np.isfinite(null["fc"]) & np.isfinite(null["beta"])   # perms where BOTH features succeeded
        T = null["fc"][both] - null["beta"][both]            # paired null differences T_i = fc(i) - beta(i)
        n_pairs = T.size                                     # number of valid paired permutations
        if n_pairs:                                          # need paired perms to run this test
            T_obs = true["fc"] - true["beta"]                # observed FC-beta gap
            p_greater = (np.sum(T >= T_obs) + 1) / (n_pairs + 1)          # preregistered one-sided: FC > beta
            p_less    = (np.sum(T <= T_obs) + 1) / (n_pairs + 1)          # opposite tail (beta > fc)
            p_two     = min(1.0, 2.0 * min(p_greater, p_less))            # exploratory two-sided (tail doubling)
            print(f"[fc-beta] T_obs={T_obs:+.4f} | null {T.mean():+.4f}±{T.std():.4f} | n={n_pairs}")   # log
            print(f"          one-sided p(FC>beta)={p_greater:.4f} | two-sided p={p_two:.4f}")           # log
            w.writerow(["fc_minus_beta_1sided_fc_gt_beta", f"{T_obs:+.4f}", n_pairs,          # preregistered row
                        f"{T.mean():+.4f}", f"{T.std():.4f}", f"{p_greater:.4f}", f"{1/(n_pairs+1):.4f}"])
            w.writerow(["fc_minus_beta_2sided_exploratory", f"{T_obs:+.4f}", n_pairs,         # exploratory row
                        f"{T.mean():+.4f}", f"{T.std():.4f}", f"{p_two:.4f}", f"{2/(n_pairs+1):.4f}"])

    print(f"\nWrote {out_csv}")                               # done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simple parallel permutation test (FC vs beta).")   # CLI parser
    ap.add_argument("--n_perms", type=int, default=N_PERMS_DEFAULT, help="permutations per feature type")   # --n_perms
    ap.add_argument("--n_jobs",  type=int, default=N_JOBS_DEFAULT,  help="parallel workers (-1 = all cores)")  # --n_jobs
    args = ap.parse_args()                                    # read the command-line arguments
    run(args.n_perms, args.n_jobs)                            # go