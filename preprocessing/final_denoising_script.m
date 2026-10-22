% =========================================================================
% CONN DENOISING  (HC / MNC sample, conn_project01)
% -------------------------------------------------------------------------
% Applies the preregistered denoising pipeline:
%    White Matter  5 aCompCor components + 1st derivative
%    CSF           5 aCompCor components + 1st derivative
%    realignment   6 motion parameters   + 1st derivative
%    = 32 confound regressors
%    no detrending | despiking after regression | bandpass 0.01-0.1 Hz (RegBP)
%
% Two-stage design: conn_batch only runs the confound-normalisation routine
% when Denoising.done is 0, so the structure is set first, inspected, then
% run. The last step is done by hand in the GUI (see the note at the end).
% =========================================================================
clear; clc;

% project_paths.m lives at the repository root. Add it to the MATLAB path
% only if it is not already there; in the repo it sits one level above this
% script, so the relative form works in a fresh clone. Must come after CLEAR.
if exist('project_paths', 'file') ~= 2
    addpath(fileparts(fileparts(mfilename('fullpath'))));
end
p = project_paths();

addpath(p.spm_dir);
spm('defaults','fmri');

batch.filename   = p.conn_project_mat;
batch.parallel.N = 6;

batch.Denoising.filter      = [0.01, 0.1];
batch.Denoising.detrending  = 0;

batch.Denoising.despiking   = 2;
batch.Denoising.regbp       = 1;
batch.Denoising.overwrite   = 1;

batch.Denoising.confounds.names      = {'White Matter','CSF','realignment'};
batch.Denoising.confounds.dimensions = {5, 5, 6};
batch.Denoising.confounds.deriv      = {1, 1, 1};
batch.Denoising.confounds.power      = {1, 1, 1};
batch.Denoising.confounds.filter     = {0, 0, 0};

batch.Denoising.done = 0;    % sets the structure
conn_batch(batch);

global CONN_x;
CONN_x.Preproc.confounds     % check: deriv {1 1 1}, dimensions {[5 16] [5 16] [6 6]}

batch.Denoising.done = 1;    % then run
conn_batch(batch);

% MANUAL STEP: go back to the GUI, on the preprocessing page click "done" and
% choose "do not overwrite". It loads for a few seconds, then the denoising
% tab appears. This script does not run start to finish on its own.