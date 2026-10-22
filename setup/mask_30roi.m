% =========================================================================
% BUILD 30-ROI BRAINNETOME MASK  (run once, before the ROI beta-map GLM)
% -------------------------------------------------------------------------
% Creates a binary mask that is 1 inside your 30 Brainnetome ROIs and 0
% elsewhere, in the same 2 mm MNI space as the atlas / functionals. Use this
% file as the explicit mask in the GLM to restrict beta estimation to the ROIs.
% =========================================================================
clear; clc;
spm('defaults', 'FMRI');

% project_paths.m lives at the repository root. Add it to the MATLAB path
% only if it is not already there; in the repo it sits one level above this
% script, so the relative form works in a fresh clone. Must come after CLEAR.
if exist('project_paths', 'file') ~= 2
    addpath(fileparts(fileparts(mfilename('fullpath'))));
end
p = project_paths();

atlas_path    = p.atlas_246;
roi_mask_path = p.roi30_mask;

% The 30 Brainnetome region indices used in the ROI FC analysis
roi_ids = [211 212 108 139 35 1 42 105 173 80 206 33 9 221 151 141 97 107 ...
           205 32 18 51 2 52 37 64 216 213 38 178];
roi_ids = unique(roi_ids);   % drop accidental duplicates

% --- read atlas and build mask ---
Vat = spm_vol(atlas_path);
A   = spm_read_vols(Vat);
maskvol = ismember(round(A), roi_ids);

% --- sanity report ---
present = unique(round(A(ismember(round(A), roi_ids))))';
missing = setdiff(roi_ids, present);
fprintf('Requested ROIs : %d\n', numel(roi_ids));
fprintf('Found in atlas : %d\n', numel(present));
fprintf('Voxels in mask : %d\n', nnz(maskvol));
if ~isempty(missing)
    warning('These ROI indices were NOT found in the atlas: %s', mat2str(missing));
end

% --- write mask (uint8, same header/space as atlas) ---
Vout = Vat;
Vout.fname = roi_mask_path;
Vout.dt    = [spm_type('uint8') 0];
spm_write_vol(Vout, double(maskvol));
fprintf('Saved mask -> %s\n', roi_mask_path);