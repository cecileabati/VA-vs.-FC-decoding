function p = project_paths()
% PROJECT_PATHS  Central path configuration for the VA/FC decoding project.
%
%   p = project_paths() returns a struct whose fields are every absolute path
%   the MATLAB scripts need. No server path appears in version control: only
%   the roots live in paths.json, which is gitignored, and the layout below
%   them is fixed and committed.
%
%   Example
%       p = project_paths();
%       base_data_dir   = p.sample_dir;
%       atlas_path      = p.atlas_246;
%       output_base_dir = p.beta_whole_brain;
%       conn_data_dir   = p.conn_data_dir;
%
%   paths.json is looked for in this order:
%       1. $VAFC_PATHS            (full path to the file)
%       2. paths.json             (next to this file)
%       3. config/paths.json
%       4. ~/.vafc_paths.json
%
%   project_paths('check') prints every path and whether it exists.
%
%   Requires MATLAB R2016b or newer for jsondecode.

    cfgfile = locate_config();
    cfg = jsondecode(fileread(cfgfile));

    if ~isfield(cfg, 'vault_root')
        error('project_paths:noRoot', ...
              'paths.json must define "vault_root" (see paths.example.json)');
    end

    p.config_file = cfgfile;
    p.vault_root  = expand_tilde(cfg.vault_root);

    % conn_base may be absolute, relative to vault_root, or absent
    if isfield(cfg, 'conn_base') && ~isempty(cfg.conn_base)
        cb = cfg.conn_base;
    else
        cb = 'new_conn_preprocessing_160725';
    end
    if is_abs(cb)
        p.conn_base = expand_tilde(cb);
    else
        p.conn_base = fullfile(p.vault_root, cb);
    end

    if isfield(cfg, 'perm_scratch') && ~isempty(cfg.perm_scratch)
        p.perm_scratch = expand_tilde(cfg.perm_scratch);
    else
        p.perm_scratch = fullfile(tempdir, 'photon_perms');
    end

    % ---- fixed layout under vault_root
    p.sample_dir   = fullfile(p.vault_root, 'sample');
    p.atlas_246    = fullfile(p.vault_root, 'BN_Atlas_246_2mm_MRICroGL.nii');
    p.spm_dir      = fullfile(p.vault_root, 'spm');
    p.conn_toolbox = fullfile(p.vault_root, 'NITRC-multi-file-downloads', 'conn');

    % ---- fixed layout under conn_base
    p.photonai_dir   = fullfile(p.conn_base, 'PhotonAI');
    p.mapping_csv    = fullfile(p.photonai_dir, 'conn_subject_mapping.csv');
    p.project_folder = fullfile(p.photonai_dir, 'Test_script_combined');

    p.beta_dir         = fullfile(p.conn_base, 'beta_maps');
    p.beta_whole_brain = fullfile(p.beta_dir, 'Beta_maps_brainnetome_32conf');
    p.beta_roi30       = fullfile(p.beta_dir, 'Beta_maps_brainnetome_30ROI_32conf');
    p.roi30_mask       = fullfile(p.beta_dir, 'roi30_mask.nii');

    p.conn_project_mat = fullfile(p.conn_base, 'conn_project01.mat');
    p.conn_data_dir    = fullfile(p.conn_base, 'conn_project01', 'data');

    firstlevel         = fullfile(p.conn_base, 'conn_project01', 'results', 'firstlevel');
    p.fc_whole_brain   = fullfile(firstlevel, 'Brainnetome');
    p.fc_roi30         = fullfile(firstlevel, 'Brainnetome_30ROIs');
end


function f = locate_config()
    here = fileparts(mfilename('fullpath'));
    cand = { getenv('VAFC_PATHS'), ...
             fullfile(here, 'paths.json'), ...
             fullfile(here, 'config', 'paths.json'), ...
             fullfile(home_dir(), '.vafc_paths.json') };
    for k = 1:numel(cand)
        if ~isempty(cand{k}) && exist(cand{k}, 'file') == 2
            f = cand{k};
            return
        end
    end
    msg = sprintf(['No paths.json found. Looked in:\n']);
    for k = 1:numel(cand)
        if ~isempty(cand{k})
            msg = [msg sprintf('    %s\n', cand{k})]; %#ok<AGROW>
        end
    end
    msg = [msg sprintf(['\nCopy paths.example.json to paths.json, put your own ' ...
                        'roots in it, and keep\nit out of git. Or point ' ...
                        '$VAFC_PATHS at it.\n'])];
    error('project_paths:noConfig', '%s', msg);
end


function tf = is_abs(s)
    tf = ~isempty(s) && (s(1) == '/' || s(1) == '~' || ...
         (numel(s) > 2 && s(2) == ':'));
end


function s = expand_tilde(s)
    if ~isempty(s) && s(1) == '~'
        s = fullfile(home_dir(), s(2:end));
    end
end


function h = home_dir()
% Portable, and works under "matlab -nojvm" where java calls are unavailable.
    h = getenv('HOME');
    if isempty(h)
        h = getenv('USERPROFILE');      % Windows
    end
    if isempty(h)
        h = pwd;
    end
end