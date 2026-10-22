#!/usr/bin/env python3
"""Central path configuration for the VA/FC decoding project.

Every absolute path used by the Python scripts resolves through this module,
so no server path appears anywhere in version control. Only the ROOTS live in
paths.json, which is gitignored; the layout below them is fixed and is
committed, because it documents what the repository expects to find.

    from project_paths import PROJECT_FOLDER, MAPPING_CSV, for_scope
    fc_dir, beta_dir = for_scope("whole_brain")

paths.json is looked for in this order:

    1. $VAFC_PATHS                       (full path to the file)
    2. ./paths.json                      (next to this module)
    3. ./config/paths.json
    4. ~/.vafc_paths.json

Run this file directly to check a configuration:

    python project_paths.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

_SEARCH = [
    os.environ.get("VAFC_PATHS"),
    os.path.join(_HERE, "paths.json"),
    os.path.join(_HERE, "config", "paths.json"),
    os.path.expanduser("~/.vafc_paths.json"),
]

_HELP = """
No paths.json found. Looked in:
{tried}

Copy paths.example.json to paths.json, put your own roots in it, and keep it
out of git (it is already in .gitignore). Or point $VAFC_PATHS at it:

    export VAFC_PATHS=/somewhere/private/paths.json
"""


def _load():
    tried = [p for p in _SEARCH if p]
    for p in tried:
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh), p
    raise FileNotFoundError(
        _HELP.format(tried="\n".join("    " + t for t in tried)))


_cfg, CONFIG_FILE = _load()

if "vault_root" not in _cfg:
    raise KeyError("paths.json must define 'vault_root' (see paths.example.json)")

# ----------------------------------------------------------------- the roots

VAULT_ROOT = os.path.expanduser(_cfg["vault_root"])

# conn_base may be absolute, or relative to vault_root, or left out entirely
_conn = _cfg.get("conn_base") or "new_conn_preprocessing_160725"
CONN_BASE = _conn if os.path.isabs(_conn) else os.path.join(VAULT_ROOT, _conn)

# node-local scratch for per-permutation PHOTONAI logging, never the shared vault
PERM_SCRATCH = os.path.expanduser(
    os.environ.get("PERM_PROJECT_FOLDER",
                   _cfg.get("perm_scratch", "/tmp/photon_perms")))

# ------------------------------------------------- fixed layout under vault_root

SAMPLE_DIR   = os.path.join(VAULT_ROOT, "sample")
ATLAS_246    = os.path.join(VAULT_ROOT, "BN_Atlas_246_2mm_MRICroGL.nii")
SPM_DIR      = os.path.join(VAULT_ROOT, "spm")
CONN_TOOLBOX = os.path.join(VAULT_ROOT, "NITRC-multi-file-downloads", "conn")

# -------------------------------------------------- fixed layout under conn_base

PHOTONAI_DIR   = os.path.join(CONN_BASE, "PhotonAI")
MAPPING_CSV    = os.path.join(PHOTONAI_DIR, "conn_subject_mapping.csv")
PROJECT_FOLDER = os.path.join(PHOTONAI_DIR, "Test_script_combined")

BETA_DIR         = os.path.join(CONN_BASE, "beta_maps")
BETA_WHOLE_BRAIN = os.path.join(BETA_DIR, "Beta_maps_brainnetome_32conf")
BETA_ROI30       = os.path.join(BETA_DIR, "Beta_maps_brainnetome_30ROI_32conf")
ROI30_MASK       = os.path.join(BETA_DIR, "roi30_mask.nii")

CONN_PROJECT_MAT = os.path.join(CONN_BASE, "conn_project01.mat")
CONN_DATA_DIR    = os.path.join(CONN_BASE, "conn_project01", "data")

_FIRSTLEVEL    = os.path.join(CONN_BASE, "conn_project01", "results", "firstlevel")
FC_WHOLE_BRAIN = os.path.join(_FIRSTLEVEL, "Brainnetome")
FC_ROI30       = os.path.join(_FIRSTLEVEL, "Brainnetome_30ROIs")

# results written by the decoding script
FAIL_DIR = os.path.join(PROJECT_FOLDER, "failures")
CKPT_DIR = os.path.join(PROJECT_FOLDER, "checkpoints")


def for_scope(scope):
    """Return (fc_folder, beta_folder) for 'whole_brain' or 'roi'."""
    if scope == "whole_brain":
        return FC_WHOLE_BRAIN, BETA_WHOLE_BRAIN
    if scope == "roi":
        return FC_ROI30, BETA_ROI30
    raise ValueError("Unknown SCOPE %r (expected 'whole_brain' or 'roi')" % scope)


def ensure_output_dirs():
    """Create the folders the decoding script writes into."""
    for d in (PROJECT_FOLDER, PERM_SCRATCH, FAIL_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)


_INPUTS = [
    ("VAULT_ROOT", VAULT_ROOT), ("CONN_BASE", CONN_BASE),
    ("SAMPLE_DIR", SAMPLE_DIR), ("ATLAS_246", ATLAS_246),
    ("SPM_DIR", SPM_DIR), ("CONN_TOOLBOX", CONN_TOOLBOX),
    ("MAPPING_CSV", MAPPING_CSV),
    ("BETA_WHOLE_BRAIN", BETA_WHOLE_BRAIN), ("BETA_ROI30", BETA_ROI30),
    ("ROI30_MASK", ROI30_MASK),
    ("CONN_PROJECT_MAT", CONN_PROJECT_MAT), ("CONN_DATA_DIR", CONN_DATA_DIR),
    ("FC_WHOLE_BRAIN", FC_WHOLE_BRAIN), ("FC_ROI30", FC_ROI30),
]
_OUTPUTS = [
    ("PROJECT_FOLDER", PROJECT_FOLDER), ("PERM_SCRATCH", PERM_SCRATCH),
    ("FAIL_DIR", FAIL_DIR), ("CKPT_DIR", CKPT_DIR),
]


def check():
    """Print every resolved path and whether it exists. Returns missing inputs."""
    print("config file : %s\n" % CONFIG_FILE)
    missing = []
    for title, group, required in (("inputs", _INPUTS, True),
                                   ("outputs (created on demand)", _OUTPUTS, False)):
        print("--- %s" % title)
        for name, p in group:
            ok = os.path.exists(p)
            print("  %-17s %s  %s" % (name, "OK     " if ok else "MISSING", p))
            if required and not ok:
                missing.append((name, p))
        print()
    if missing:
        print("%d input path(s) missing. Fix the roots in %s, or check that this"
              % (len(missing), CONFIG_FILE))
        print("machine actually has the data mounted.")
    else:
        print("all input paths resolve.")
    return missing


if __name__ == "__main__":
    sys.exit(1 if check() else 0)