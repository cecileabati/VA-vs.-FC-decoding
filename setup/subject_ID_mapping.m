%% CONN subject mapping extraction

clear; clc;

% project_paths.m lives at the repository root. Add it to the MATLAB path
% only if it is not already there; in the repo it sits one level above this
% script, so the relative form works in a fresh clone. Must come after CLEAR.
if exist('project_paths', 'file') ~= 2
    addpath(fileparts(fileparts(mfilename('fullpath'))));
end
p = project_paths();

% CONN project folder and the project file inside it
projdir     = p.conn_base;
projectfile = p.conn_project_mat;

% If the expected file is not found, list available .mat files
if ~exist(projectfile, 'file')
    fprintf('Could not find:\n%s\n\n', projectfile);
    fprintf('Available .mat files in project folder:\n');
    disp(dir(fullfile(projdir, '*.mat')));
    error('Please check the exact .mat filename and update paths.json.');
end

% Load CONN project
load(projectfile, 'CONN_x');

% Number of subjects in the CONN project
nsub = CONN_x.Setup.nsubjects;

% Prepare output variables
conn_subject     = (1:nsub)';
original_id      = cell(nsub,1);
functional_file  = cell(nsub,1);
structural_file  = cell(nsub,1);

for s = 1:nsub

    %% -----------------------------
    % Functional file, Session 1
    %% -----------------------------
    func = CONN_x.Setup.functional{s}{1};

    % Unwrap nested cells
    while iscell(func)
        func = func{1};
    end

    % If func is a char matrix with one row per volume, take first volume
    if ischar(func) && size(func,1) > 1
        func = strtrim(func(1,:));
    end

    % Remove volume index, e.g. ",1"
    func = regexprep(func, ',\d+$', '');

    functional_file{s} = func;

    %% -----------------------------
    % Structural file
    %% -----------------------------
    anat = CONN_x.Setup.structural{s};

    % Unwrap nested cells
    while iscell(anat)
        anat = anat{1};
    end

    % If anat is a char matrix, take first row
    if ischar(anat) && size(anat,1) > 1
        anat = strtrim(anat(1,:));
    end

    % Remove volume index, e.g. ",1"
    anat = regexprep(anat, ',\d+$', '');

    structural_file{s} = anat;

    %% -----------------------------
    % Extract original subject ID
    %
    % This matches:
    % JN001
    % JN002
    % JN015
    % N012
    % N089
    % N091_prae
    % N098_prae
    %% -----------------------------
    id = regexp(func, '(JN\d{3}|N\d{3}(?:_(?:prae|\d+J|J\d+))?)', 'match', 'once');

    % If not found in functional path, try structural path
    if isempty(id)
        id = regexp(anat,'(JN\d{3}|N\d{3}(?:_(?:prae|J\d+|\d+J))?)','match', 'once');
    end

    original_id{s} = id;
end

% Create mapping table
T = table(conn_subject, original_id, functional_file, structural_file);

% Display CONN subject number + original ID
disp(T(:, {'conn_subject', 'original_id'}));

% Save CSV where the decoding pipeline expects to read it
outfile = p.mapping_csv;
if ~exist(fileparts(outfile), 'dir')
    mkdir(fileparts(outfile));
end
writetable(T, outfile);

fprintf('\nMapping saved to:\n%s\n', outfile);