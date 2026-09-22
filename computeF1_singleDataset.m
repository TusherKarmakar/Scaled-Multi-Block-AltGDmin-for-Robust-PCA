function computeF1_singleDataset(datasetOutFolder, datasetRootFolder, resultsOutFolder)
%COMPUTEF1_SINGLEDATASET F1 evaluation for one dataset, using the EXACT
% same alpha-based thresholding step used during training
% (run_batch_altgdmin_main.m): estimate_alpha.m + thresh_T_alpha_vec.m,
% with the same tau_abs/c_norm/sigma_min/c_corr/gamma/batchSize values -
% read directly from THIS dataset's own simulation_parameters.txt, so it
% can never drift out of sync with however the pipeline was actually run.
%
% datasetOutFolder  - where background.mat / simulation_parameters.txt
%                      already live (READ-ONLY source, from the training run)
% datasetRootFolder - CDnet dataset root (input/, groundtruth/, ...)
% resultsOutFolder  - where THIS function's outputs are written
%                      (f1_results.txt and the video) - separate from
%                      datasetOutFolder, so the original training output
%                      folder is never touched
%
% DETECTION RULE (standard "S_hat ~= 0" support-recovery rule):
%   background_batch = U*B + mean_b               (already saved on disk)
%   R_batch = Y_batch - background_batch = Yc_b - U*B   <- exactly what
%             thresh_T_alpha_vec saw on training's LAST iteration
%   gaRow/gaCol recomputed the same way training computed them:
%       [alpha_row, alpha_col] = estimate_alpha(Yc_b, tau_abs, c_norm, sigma_min)
%       alpha_row = c_corr * alpha_row;  alpha_col = c_corr * alpha_col;
%       gaRow = min(gamma*alpha_row, 1); gaCol = min(gamma*alpha_col, 1);
%   S_hat  = thresh_T_alpha_vec(R_batch, gaRow, gaCol)
%   foreground mask = (S_hat ~= 0)
% Since alpha estimation is a deterministic function of Yc_b, this
% reproduces training's actual final S for that batch exactly.
%
% SCORING RULE: matches the official CDnet comparator EXACTLY
% (compareImageFiles.m / compare(), Nil Goyette, U. Sherbrooke):
%   TP = (gt==255 & pred==fg),  FP = (gt<=50 & pred==fg)
%   FN = (gt==255 & pred==bg),  TN = (gt<=50 & pred==bg)
%   SE = (gt==50  & pred==fg)   <- Shadow Error, diagnostic only (subset of FP)
% Note gt<=50 covers BOTH static (0) and shadow (50) as background - shadow
% is NOT excluded, it's folded into background per the official rule. GT
% values 85 (outside ROI) and 170 (unknown motion) are excluded automatically
% since neither condition matches them - no separate ROI.bmp masking is used
% (the official comparator does not apply ROI.bmp either).
%
% NOTE: this function calls estimate_alpha.m and thresh_T_alpha_vec.m
% directly (same ones used in run_batch_altgdmin_main.m), and reimplements
% the official compare() logic above inline (rather than calling
% processVideoFolder.m, which expects pre-saved binary mask PNG files on
% disk - here the mask is computed in-memory per batch instead).
%
% Writes into resultsOutFolder:
%   f1_results.txt                       - Precision/Recall/F1/ShadowError/TimePerFrame
%   <category>_<dataset>_orig_bg_fg.mp4  - video: original | background | foreground mask, side by side

    if ~exist(resultsOutFolder, 'dir')
        mkdir(resultsOutFolder);
    end

    %% ---- Locate this dataset ----
    [~, folderName] = fileparts(datasetOutFolder);
    folderName = char(folderName);   % guard against datasetOutFolder being a "string" type,
                                      % where folderName == '_' would compare whole strings
                                      % instead of individual characters
    underscoreIdx = find(folderName == '_', 1, 'first');
    if isempty(underscoreIdx)
        error('Folder name "%s" is not "<category>_<dataset>".', folderName);
    end
    category = folderName(1:underscoreIdx-1);
    dataset  = folderName(underscoreIdx+1:end);

    seqFolder   = fullfile(datasetRootFolder, category, dataset);
    inputFolder = fullfile(seqFolder, 'input');
    gtFolder    = fullfile(seqFolder, 'groundtruth');

    %% ---- Read THIS dataset's exact parameters from simulation_parameters.txt ----
    paramsPath = fullfile(datasetOutFolder, 'simulation_parameters.txt');
    if ~exist(paramsPath, 'file')
        error('simulation_parameters.txt not found in %s', datasetOutFolder);
    end
    ptxt = fileread(paramsPath);

    tau_abs      = readParam(ptxt, 'tau_abs');
    c_norm       = readParam(ptxt, 'c_norm');
    sigma_min    = readParam(ptxt, 'sigma_min');
    c_corr       = readParam(ptxt, 'c_corr \(alpha correction\)');
    gamma        = readParam(ptxt, 'gamma');
    batchSize    = readParam(ptxt, 'batchSize');
    timePerFrame = readParam(ptxt, 'time_per_frame_sec');

    %% ---- Locate background source: dense vs compact ----
    denseMatPath   = fullfile(datasetOutFolder, 'background.mat');
    compactMatPath = fullfile(datasetOutFolder, 'background_UB.mat');

    if exist(denseMatPath, 'file')
        matObj  = matfile(denseMatPath);
        bgInfo  = whos(matObj, 'background');
        q_total = bgInfo.size(2);
        getBgBlock = @(startIdx, endIdx) matObj.background(:, startIdx:endIdx);
    elseif exist(compactMatPath, 'file')
        error(['background_UB.mat (compact) found for %s, but this function ' ...
               'does not yet know its variable names (U/B/mean field names). ' ...
               'Tell me exactly how it is saved and I will wire up reconstruction.'], folderName);
    else
        error('Neither background.mat nor background_UB.mat found in %s', datasetOutFolder);
    end

    %% ---- Frame list, GT, temporalROI ----
    inFiles = dir(fullfile(inputFolder, 'in*.jpg'));
    inFiles = sort_nat_by_number(inFiles);

    gtFiles = dir(fullfile(gtFolder, 'gt*.png'));
    gtFiles = sort_nat_by_number(gtFiles);

    nFrames = min([q_total, numel(inFiles), numel(gtFiles)]);
    if nFrames < q_total || nFrames < numel(inFiles) || nFrames < numel(gtFiles)
        warning('%s: frame count mismatch (background=%d, input=%d, gt=%d) - using %d frames.', ...
            folderName, q_total, numel(inFiles), numel(gtFiles), nFrames);
    end

    I0 = imread(fullfile(inputFolder, inFiles(1).name));
    if size(I0,3) == 3
        I0 = rgb2gray(I0);
    end
    [rows, cols] = size(I0);
    n = rows * cols;

    troiPath = fullfile(seqFolder, 'temporalROI.txt');
    if exist(troiPath, 'file')
        troiText = fileread(troiPath);
        troiNums = regexp(troiText, '\d+', 'match');
        if numel(troiNums) < 2
            error('temporalROI.txt for %s does not contain two numbers (got: "%s")', ...
                folderName, strtrim(troiText));
        end
        startFrameEval = str2double(troiNums{1});
        endFrameEval   = min(str2double(troiNums{2}), nFrames);
    else
        startFrameEval = 1;
        endFrameEval   = nFrames;
        warning('%s: temporalROI.txt not found, evaluating all frames.', folderName);
    end

    %% ---- Video writer (written to resultsOutFolder, NOT datasetOutFolder) ----
    videoPath = fullfile(resultsOutFolder, [folderName '_orig_bg_fg.mp4']);
    vw = VideoWriter(videoPath, 'MPEG-4');
    vw.FrameRate = 25;
    open(vw);

    TP = 0; FP = 0; FN = 0; TN = 0; SE = 0;

    %% ---- Process in the SAME batches training used ----
    K = ceil(nFrames / batchSize);
    for b = 1:K
        startIdx = (b-1)*batchSize + 1;
        endIdx   = min(b*batchSize, nFrames);
        q_b      = endIdx - startIdx + 1;

        % --- read this batch's raw frames (same as training) ---
        Y_b = zeros(n, q_b);
        for jj = 1:q_b
            Ij = imread(fullfile(inputFolder, inFiles(startIdx+jj-1).name));
            if size(Ij,3) == 3
                Ij = rgb2gray(Ij);
            end
            Y_b(:,jj) = double(Ij(:));
        end

        mean_b = mean(Y_b, 2);
        Yc_b   = Y_b - mean_b;

        % --- SAME alpha estimation as training ---
        [alpha_row, alpha_col] = estimate_alpha(Yc_b, tau_abs, c_norm, sigma_min);
        alpha_row = c_corr * alpha_row;
        alpha_col = c_corr * alpha_col;
        gaRow = min(gamma * alpha_row, 1);
        gaCol = min(gamma * alpha_col, 1);

        % --- this batch's saved background block ---
        Bg_b = getBgBlock(startIdx, endIdx);

        % --- SAME thresholding step as training's final S (S_hat ~= 0 rule) ---
        R_b      = Y_b - Bg_b;                          % = Yc_b - U*B, exactly
        S_hat    = thresh_T_alpha_vec(R_b, gaRow, gaCol);
        fgMask_b = (S_hat ~= 0);

        for jj = 1:q_b
            i = startIdx + jj - 1;

            frameGray = reshape(Y_b(:,jj),   rows, cols);
            bgFrame   = reshape(Bg_b(:,jj),  rows, cols);
            fgFrame   = reshape(fgMask_b(:,jj), rows, cols);

            if i >= startFrameEval && i <= endFrameEval
                gtImg = imread(fullfile(gtFolder, gtFiles(i).name));
                if size(gtImg,3) == 3
                    gtImg = rgb2gray(gtImg);
                end

                % EXACT official CDnet comparator formula (compareImageFiles.m /
                % compare() by Nil Goyette, U. Sherbrooke): GT<=50 counts as
                % background (this INCLUDES shadow=50, not just static=0);
                % GT==255 is foreground; GT values 85 (outside ROI) and 170
                % (unknown motion) are excluded automatically since neither
                % condition matches them. No separate ROI.bmp masking is used.
                TP = TP + nnz(gtImg==255 & fgFrame);
                FP = FP + nnz(gtImg<=50  & fgFrame);
                FN = FN + nnz(gtImg==255 & ~fgFrame);
                TN = TN + nnz(gtImg<=50  & ~fgFrame);
                SE = SE + nnz(gtImg==50  & fgFrame);   % Shadow Error (diagnostic only, subset of FP)
            end

            % --- write video frame: original | background | foreground ---
            origRGB = repmat(uint8(frameGray), 1, 1, 3);
            bgRGB   = repmat(uint8(max(0, min(255, bgFrame))), 1, 1, 3);
            fgRGB   = repmat(uint8(fgFrame) * 255, 1, 1, 3);
            writeVideo(vw, [origRGB, bgRGB, fgRGB]);
        end
    end

    close(vw);

    nEvalFrames = endFrameEval - startFrameEval + 1;
    fprintf('    frames evaluated: %d (startFrameEval=%d, endFrameEval=%d, nFrames=%d)\n', ...
        nEvalFrames, startFrameEval, endFrameEval, nFrames);
    fprintf('    TP=%d  FP=%d  FN=%d  TN=%d  SE(shadow error)=%d\n', TP, FP, FN, TN, SE);

    if nEvalFrames <= 0
        warning(['%s: evaluated frame range is EMPTY (startFrameEval=%d > endFrameEval=%d). ' ...
            'Check temporalROI.txt for this sequence - it likely parsed incorrectly ' ...
            '(expects two integers "startFrame endFrame").'], folderName, startFrameEval, endFrameEval);
    elseif (TP+FP+FN+TN) == 0
        warning(['%s: zero valid pixels were counted across all %d evaluated frames. ' ...
            'Check that groundtruth/*.png exist and are readable for this sequence.'], ...
            folderName, nEvalFrames);
    elseif (TP+FP) == 0
        warning('%s: predicted foreground was empty in every evaluated frame (Precision is NaN).', folderName);
    elseif (TP+FN) == 0
        warning('%s: ground-truth foreground was empty in every evaluated frame (Recall is NaN).', folderName);
    end

    precision = TP / (TP + FP);
    recall    = TP / (TP + FN);
    f1        = 2 * precision * recall / (precision + recall);

    %% ---- Write f1_results.txt (to resultsOutFolder, NOT datasetOutFolder) ----
    resPath = fullfile(resultsOutFolder, 'f1_results.txt');
    fid = fopen(resPath, 'w');
    fprintf(fid, 'Dataset: %s/%s\n', category, dataset);
    fprintf(fid, 'Precision: %.6f\n', precision);
    fprintf(fid, 'Recall: %.6f\n', recall);
    fprintf(fid, 'F1: %.6f\n', f1);
    fprintf(fid, 'ShadowError: %d\n', SE);
    fprintf(fid, 'TimePerFrame: %.6f\n', timePerFrame);
    fclose(fid);

    fprintf('  %s/%s: P=%.4f R=%.4f F1=%.4f  time/frame=%.4f s\n', ...
        category, dataset, precision, recall, f1, timePerFrame);
end

%% ================= Local functions =================
function files = sort_nat_by_number(files)
    % Sorts a dir() struct array by the numeric part of each filename
    % (e.g. in000001.jpg / gt000001.png) rather than lexicographically.
    names = {files.name};
    nums = zeros(numel(names),1);
    for k = 1:numel(names)
        digs = regexp(names{k}, '\d+', 'match');
        nums(k) = str2double(digs{end});
    end
    [~, idx] = sort(nums);
    files = files(idx);
end

function val = readParam(txt, name)
    tok = regexp(txt, [name '\s*:\s*([\d.eE+\-]+)'], 'tokens', 'once');
    if isempty(tok)
        error('Could not find parameter "%s" in simulation_parameters.txt', name);
    end
    val = str2double(tok{1});
end
