% =========================================================================
% MASTER SCRIPT: BETA MAPS  (whole-brain OR 30-ROI, set by SCOPE)
% -------------------------------------------------------------------------
% Finds 'wau*NewFaces.nii', extracts ID, runs GLM with 32 nuisance
% regressors (motion + WM/CSF aCompCor + derivatives), matching the CONN
% denoising used in the FC arm.
%
% SCOPE = 'whole_brain' -> mask is the full 246-region Brainnetome atlas
% SCOPE = 'roi'         -> mask is roi30_mask.nii (run mask_30roi.m first)
% =========================================================================

clear; clc;
spm('defaults', 'FMRI');
spm_jobman('initcfg');

SCOPE = 'whole_brain';   % 'whole_brain' or 'roi'

% -------------------------------------------------------------------------
% 1. DEFINE PATHS
% -------------------------------------------------------------------------
% project_paths.m lives at the repository root. Add it to the MATLAB path
% only if it is not already there; in the repo it sits one level above this
% script, so the relative form works in a fresh clone. Must come after CLEAR.
if exist('project_paths', 'file') ~= 2
    addpath(fileparts(fileparts(mfilename('fullpath'))));
end
p = project_paths();

base_data_dir = p.sample_dir;
atlas_path    = p.atlas_246;
roi_mask_path = p.roi30_mask;
mapping_csv   = p.mapping_csv;
conn_data_dir = p.conn_data_dir;

% --- SCOPE switch: sets the explicit mask and the output folder ---
switch SCOPE
    case 'whole_brain'
        mask_path       = atlas_path;
        output_base_dir = p.beta_whole_brain;
    case 'roi'
        mask_path       = roi_mask_path;
        output_base_dir = p.beta_roi30;
        if ~exist(roi_mask_path, 'file')
            error('roi30_mask.nii not found - run mask_30roi.m first.');
        end
    otherwise
        error('Unknown SCOPE ''%s'' - use ''whole_brain'' or ''roi''.', SCOPE);
end

fprintf('SCOPE  : %s\n', SCOPE);
fprintf('mask   : %s\n', mask_path);
fprintf('output : %s\n', output_base_dir);

%% -------------------------------------------------------------------------
% 2. FIND FILES
% -------------------------------------------------------------------------
fprintf('Scanning for wau*NewFaces.nii files...\n');
file_list = dir(fullfile(base_data_dir, '**', 'wau*NewFaces.nii'));

if isempty(file_list)
    error('No files matching wau*NewFaces.nii were found!');
end

fprintf('Found %d functional files.\n', length(file_list));

% --- Subject mapping (read once, before the loop) ---
% readtable's auto-detection splits the file-path columns on '/', so parse
% the CSV explicitly: one number + three strings per row.
fid = fopen(mapping_csv);
fgetl(fid);                                              % discard header line
Craw = textscan(fid, '%f%s%s%s', 'Delimiter', ',');
fclose(fid);

conn_nums  = Craw{1};
orig_ids   = strtrim(string(Craw{2}));
func_files = strtrim(string(Craw{3}));

fprintf('Loaded mapping for %d CONN subjects.\n', numel(conn_nums));

chk = orig_ids(conn_nums == 309);
if ~isempty(chk) && chk ~= "N439_prae"
    warning('conn_subject 309 maps to %s, not N439_prae', chk);
end

%% -------------------------------------------------------------------------
% 3. LOOP
% -------------------------------------------------------------------------
for i = 1:length(file_list)

    % --- A. Extract ID ---
    nifti_name    = file_list(i).name;   % e.g. "wauN460_prae_NewFaces.nii"
    subj_func_dir = file_list(i).folder;

    % STRICT REGEX: Only accepts exactly 'wau' + ID + '_NewFaces.nii'
    tokens = regexp(nifti_name, '^wau(.+?)_NewFaces\.nii$', 'tokens');

    if isempty(tokens)
        warning('Skipping %s: Filename format does not match expected pattern.', nifti_name);
        continue;
    end

    subj_id = tokens{1}{1}; % e.g. "N460_prae"

    % --- B. Setup Output ---
    subj_output_dir = fullfile(output_base_dir, subj_id);

    fprintf('-------------------------------------------------------\n');
    fprintf('Processing %d/%d: %s\n', i, length(file_list), subj_id);

    try
        if ~exist(subj_output_dir, 'dir')
            mkdir(subj_output_dir);
        end

        % --- C. Find Motion Parameters ---
        rp_struct = dir(fullfile(subj_func_dir, 'rp_*.txt'));
        if isempty(rp_struct)
            warning('SKIPPING %s: rp_*.txt not found in %s', subj_id, subj_func_dir);
            continue;
        end
        rp_file = fullfile(rp_struct(1).folder, rp_struct(1).name);

        % --- D. Build 32-column confound file ---
        row = orig_ids == string(subj_id);
        if sum(row) ~= 1
            warning('SKIPPING %s: %d matches in mapping CSV', subj_id, sum(row));
            continue;
        end
        conn_idx = conn_nums(row);

        % cross-check: CONN's record must point at THIS file
        [~, csv_name] = fileparts(func_files(row));
        if csv_name ~= string(erase(nifti_name, '.nii'))
            warning('SKIPPING %s: CSV points at %s', subj_id, csv_name);
            continue;
        end

        C = load(fullfile(conn_data_dir, ...
                 sprintf('ROI_Subject%03d_Session001.mat', conn_idx)));

        iWM  = strcmp(C.names, 'White Matter');
        iCSF = strcmp(C.names, 'CSF');
        if sum(iWM) ~= 1 || sum(iCSF) ~= 1
            warning('SKIPPING %s: WM/CSF ROI not found in CONN file', subj_id);
            continue;
        end
        if size(C.data{iWM},2) < 5 || size(C.data{iCSF},2) < 5
            warning('SKIPPING %s: fewer than 5 components available', subj_id);
            continue;
        end

        wm   = C.data{iWM}(:,1:5);
        csf  = C.data{iCSF}(:,1:5);
        rp   = load(rp_file);

        base = [rp, wm, csf];                          % 210 x 16
        R    = [base, [zeros(1,16); diff(base)]];      % 210 x 32

        conf_file = fullfile(subj_output_dir, 'confounds32.txt');
        writematrix(R, conf_file, 'Delimiter', 'tab');

        % --- E. Select Scans ---
        scans = cellstr(spm_select('ExtFPList', subj_func_dir, ['^', nifti_name, '$'], Inf));

        if size(R,1) ~= numel(scans)
            warning('SKIPPING %s: %d confound rows vs %d scans', subj_id, size(R,1), numel(scans));
            continue;
        end

%%      % -----------------------------------------------------------------
        % 4. GLM BATCH
        % -----------------------------------------------------------------
        clear matlabbatch;

        matlabbatch{1}.spm.stats.fmri_spec.dir = {subj_output_dir};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2.1;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

        matlabbatch{1}.spm.stats.fmri_spec.sess.scans = scans;

        % Cond 1: SHAPE
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'shape';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = [2; 102; 202; 302; 402];
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 36;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 1;

        % Cond 2: NEUTRAL
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'neutral';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = [58; 75; 78; 170; 175; 186; 260; 285; 293; 340; 350; 385];
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 2;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 1;

        % Cond 3: ANGRY
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).name = 'angry';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).onset = [40; 68; 88; 150; 181; 193; 243; 248; 290; 370; 373; 388];
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).duration = 2;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).orth = 1;

        % Cond 4: HAPPY
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).name = 'happy';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).onset = [55; 81; 93; 145; 153; 156; 240; 267; 278; 355; 360; 380];
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).duration = 2;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).orth = 1;

        % Cond 5: FEARFUL
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).name = 'fearful';
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).onset = [43; 48; 63; 140; 163; 178; 255; 272; 275; 347; 363; 393];
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).duration = 2;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).orth = 1;

        % Nuisance Regressors (32 columns: motion + WM/CSF aCompCor + derivatives)
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {conf_file};

        % Global Settings
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = -Inf; % Disables the "80% mean" default of SPM
        % >>> mask set by SCOPE (full atlas or 30-ROI mask) <<<
        matlabbatch{1}.spm.stats.fmri_spec.mask = {[mask_path, ',1']};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

        % Estimation
        matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        % Run
        spm_jobman('run', matlabbatch);
        fprintf('Done with %s\n', subj_id);

    catch ME
        fprintf('ERROR processing %s: %s\n', subj_id, ME.message);
    end
end