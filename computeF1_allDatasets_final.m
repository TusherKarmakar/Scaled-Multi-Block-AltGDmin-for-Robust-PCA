%% computeF1_allDatasets.m
% Runs F1 evaluation on every dataset subfolder found inside
% outputRootFolder - no hardcoded whitelist, no function arguments.
%
% EDIT THE TWO PATHS BELOW, then just run this script.
%
% Results are written to a NEW "F1 Score" folder created wherever THIS
% SCRIPT IS RUN FROM (pwd) - NOT back into outputRootFolder. The original
% training output (background.mat, simulation_parameters.txt) is only
% ever read, never modified.
%
% For each subfolder found (expected name: "<category>_<dataset>"), this:
%   - auto-detects category/dataset from the folder name
%   - auto-detects background.mat (dense) vs background_UB.mat (compact)
%   - writes, into ./F1 Score/<category>_<dataset>/:
%       f1_results.txt            - Precision/Recall/F1/TimePerFrame
%       <name>_orig_bg_fg.mp4     - video (original | background | foreground)
%     (via computeF1_singleDataset.m, using the SAME alpha-thresholding
%     step as training - see that file's header comment)
%   - folds every dataset's F1/Recall/Precision/TimePerFrame into:
%       ./F1 Score/F1_allDatasets_summary.txt  (plain text, per-category
%                                                 averages + overall average)
%
% REQUIRES ON PATH:
%   computeF1_singleDataset.m, estimate_alpha.m, thresh_T_alpha_vec.m

clear; clc;

%% ============================================================
% CONFIG - EDIT THESE
% ============================================================
addpath('');   % <-- folder with estimate_alpha.m, thresh_T_alpha_vec.m, computeF1_singleDataset.m

outputRootFolder  = '';   % e.g. contains baseline_highway, shadow_copyMachine, ...
datasetRootFolder = '';   % folder contain dataset

% Normalize to char regardless of how they were quoted above - a "string"
% (double-quoted) type here would silently break folderName == '_'
% character comparisons later (string == compares whole strings, not
% per-character), which is what caused every dataset to fail identically.
outputRootFolder  = char(outputRootFolder);
datasetRootFolder = char(datasetRootFolder);

%% ---- Where results are actually written: pwd, not outputRootFolder ----
resultsRoot = fullfile(pwd, 'F1 Score');
if ~exist(resultsRoot, 'dir')
    mkdir(resultsRoot);
end

%% ============================================================
d = dir(outputRootFolder);
subfolders = d([d.isdir] & ~ismember({d.name}, {'.','..'}));

nTotal = numel(subfolders);
fprintf('Found %d subfolder(s) in %s\n', nTotal, outputRootFolder);
fprintf('Results will be written to: %s\n', resultsRoot);

resultsRows = cell(0, 7);   % category, dataset, precision, recall, f1, timePerFrame, status

for k = 1:nTotal
    folderName = subfolders(k).name;
    datasetOutFolder    = fullfile(outputRootFolder, folderName);
    datasetResultsFolder = fullfile(resultsRoot, folderName);

    fprintf('\n============================================================\n');
    fprintf('[%d/%d] %s\n', k, nTotal, folderName);
    fprintf('============================================================\n');

    underscoreIdx = find(folderName == '_', 1, 'first');
    if isempty(underscoreIdx)
        fprintf(2, 'SKIPPING "%s" - folder name is not "<category>_<dataset>"\n', folderName);
        resultsRows(end+1,:) = {folderName, '', NaN, NaN, NaN, NaN, 'skipped: bad folder name'}; %#ok<AGROW>
        continue;
    end
    category = folderName(1:underscoreIdx-1);
    dataset  = folderName(underscoreIdx+1:end);

    try
        computeF1_singleDataset(datasetOutFolder, datasetRootFolder, datasetResultsFolder);

        % re-read what computeF1_singleDataset just wrote, to fold into the combined summary
        resFile = fullfile(datasetResultsFolder, 'f1_results.txt');
        [precision, recall, f1, timePerFrame] = parseF1ResultsFile(resFile);

        resultsRows(end+1,:) = {category, dataset, precision, recall, f1, timePerFrame, 'ok'}; %#ok<AGROW>

    catch ME
        fprintf(2, '\nERROR on %s:\n%s\n', folderName, getReport(ME, 'extended', 'hyperlinks', 'off'));
        resultsRows(end+1,:) = {category, dataset, NaN, NaN, NaN, NaN, sprintf('error: %s', ME.message)}; %#ok<AGROW>
    end
end

okRows     = strcmp(resultsRows(:,7), 'ok');
categories = unique(resultsRows(okRows,1), 'stable');

%% ---- Plain-text summary: per-dataset rows, category averages, overall average ----
summaryFile = fullfile(resultsRoot, 'F1_allDatasets_summary.txt');
fid = fopen(summaryFile, 'wt');
fprintf(fid, 'Category\tDataset\tPrecision\tRecall\tF1\tTimePerFrame(s)\tStatus\n');

allF1   = [];
allTime = [];

for c = 1:numel(categories)
    catName = categories{c};
    idx = strcmp(resultsRows(:,1), catName) & okRows;
    catRows = resultsRows(idx,:);

    for i = 1:size(catRows,1)
        r = catRows(i,:);
        fprintf(fid, '%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%s\n', r{1}, r{2}, r{3}, r{4}, r{5}, r{6}, r{7});
    end

    catF1   = mean(cell2mat(catRows(:,5)));
    catTime = mean(cell2mat(catRows(:,6)));
    fprintf(fid, '%s\tAVERAGE\t\t\t%.6f\t%.6f\t\n', catName, catF1, catTime);

    allF1   = [allF1;   cell2mat(catRows(:,5))]; %#ok<AGROW>
    allTime = [allTime; cell2mat(catRows(:,6))]; %#ok<AGROW>
end

% list any error/skip rows too, for visibility
badIdx = ~okRows;
for i = find(badIdx)'
    r = resultsRows(i,:);
    fprintf(fid, '%s\t%s\t\t\t\t\t%s\n', r{1}, r{2}, r{7});
end

fprintf(fid, '\nOVERALL AVERAGE F1: %.6f\n', mean(allF1, 'omitnan'));
fprintf(fid, 'OVERALL AVERAGE TIME/FRAME (s): %.6f\n', mean(allTime, 'omitnan'));
fclose(fid);

fprintf('\n============================================================\n');
fprintf('ALL SUBFOLDERS DONE.\n');
fprintf('Plain summary:  %s\n', summaryFile);
fprintf('Overall F1 = %.4f, overall time/frame = %.4f s\n', mean(allF1,'omitnan'), mean(allTime,'omitnan'));
fprintf('============================================================\n');


%% ============================================================
function [precision, recall, f1, timePerFrame] = parseF1ResultsFile(resFile)
precision = NaN; recall = NaN; f1 = NaN; timePerFrame = NaN;
if ~exist(resFile, 'file')
    return;
end
fileText = fileread(resFile);
tok = regexp(fileText, 'Precision:\s*([\d.eE+\-]+)', 'tokens', 'once');
if ~isempty(tok), precision = str2double(tok{1}); end
tok = regexp(fileText, 'Recall:\s*([\d.eE+\-]+)', 'tokens', 'once');
if ~isempty(tok), recall = str2double(tok{1}); end
tok = regexp(fileText, 'F1:\s*([\d.eE+\-]+)', 'tokens', 'once');
if ~isempty(tok), f1 = str2double(tok{1}); end
tok = regexp(fileText, 'TimePerFrame:\s*([\d.eE+\-]+)', 'tokens', 'once');
if ~isempty(tok), timePerFrame = str2double(tok{1}); end
end
