%% run_batch_altgdmin_main_fast.m
clear; clc;

%% ---- Paths & datasets ----
datasetRoot = '....';    % location of root folder

datasets = { ...
    'baseline/highway', ...
    'baseline/office', ...
    'dynamicBackground/canoe', ...
    'dynamicBackground/boats', ...
    'dynamicBackground/overpass', ...
    'cameraJitter/traffic', ...
    'cameraJitter/badminton', ...
    'intermittentObjectMotion/winterDriveway', ...
    'intermittentObjectMotion/sofa', ...
    'intermittentObjectMotion/streetLight', ...
    'shadow/backdoor', ...
    'shadow/copyMachine', ...
    'shadow/cubicle', ...
    'thermal/library', ...
    'thermal/lakeSide' ...
};

%% ---- Fixed parameters (unchanged) ----
tau_abs   = 8;
c_norm    = 3.0;
sigma_min = 4.0;
c_corr = 1;
gamma  = 1.05;
energyThresh   = 0.80;
rankGrowThresh = 0.10;
batchSize = 250;
T1 = 30;
Tb = 15;

saveAsSingle = false;   % true: smaller/faster .mat writes, ~1e-7 relative rounding

outRoot = fullfile(pwd, 'Output_Final_optimized_version');
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

%% ---- Parallel pool for image reading (optional) ----
try
    if license('test','Distrib_Computing_Toolbox') && isempty(gcp('nocreate'))
        parpool;
    end
catch
    % no pool -> parfor runs serially
end

for d = 1:numel(datasets)
    dsPath  = datasets{d};
    seqName = strrep(dsPath, '/', '_');
    fprintf('=== Processing %s ===\n', dsPath);

    seqFolder   = fullfile(datasetRoot, strrep(dsPath, '/', filesep));
    inputFolder = fullfile(seqFolder, 'input');

    inFiles = dir(fullfile(inputFolder, 'in*.jpg'));
    inFiles = sort_nat_by_number(inFiles);
    q_total = numel(inFiles);
    if q_total == 0
        warning('No frames found for %s, skipping.', dsPath);
        continue;
    end
    framePaths = fullfile(inputFolder, {inFiles.name});

    I0 = imread(framePaths{1});
    if size(I0,3) == 3, I0 = rgb2gray(I0); end
    [rows, cols] = size(I0);
    n = rows * cols;

    %% ---- Output dir + on-disk background array ----
    seqOutDir = fullfile(outRoot, seqName);
    if ~exist(seqOutDir, 'dir'), mkdir(seqOutDir); end

    bgMatPath = fullfile(seqOutDir, 'background.mat');
    if exist(bgMatPath, 'file'), delete(bgMatPath); end
    matObj = matfile(bgMatPath, 'Writable', true);
    if saveAsSingle
        matObj.background(n, q_total) = single(0);
    else
        matObj.background(n, q_total) = 0;
    end

    logPath = fullfile(seqOutDir, 'batch_log.txt');
    logFid = fopen(logPath, 'w');
    fprintf(logFid, ['batch\tstartIdx\tendIdx\tq_b\tr\t' ...
        'alpha_row_mean\talpha_row_min\talpha_row_max\t' ...
        'alpha_col_mean\talpha_col_min\talpha_col_max\t' ...
        'eb\trank_grew\n']);

    K = ceil(q_total / batchSize);

    U = [];
    r = 0;
    muTimeAccum  = 0;   % mu time, excluded from reported time (as before)
    computeTime  = 0;   % timed compute (I/O excluded; initialization included)
    initTime     = 0;   % batch-1 initialization duration (informational)

    for b = 1:K
        startIdx = (b-1)*batchSize + 1;
        endIdx   = min(b*batchSize, q_total);
        q_b      = endIdx - startIdx + 1;

        %% ---- I/O (NOT timed): parallel frame read ----
        Y_b = zeros(n, q_b);
        parfor jj = 1:q_b
            Ij = imread(framePaths{startIdx + jj - 1});
            if size(Ij,3) == 3, Ij = rgb2gray(Ij); end
            Y_b(:,jj) = double(Ij(:));
        end

        %% ---- BATCH INITIALIZATION ----
        % Timer starts here for every batch, including batch 1 (init is counted).
        tc = tic;

        mean_b = mean(Y_b, 2);
        Yc_b   = Y_b - mean_b;
        clear Y_b

        [alpha_row, alpha_col] = estimate_alpha(Yc_b, tau_abs, c_norm, sigma_min);
        alpha_row = c_corr * alpha_row;
        alpha_col = c_corr * alpha_col;

        S0 = thresh_T_alpha_vec(Yc_b, alpha_row, alpha_col);

        if b == 1
            X0c = Yc_b - S0;
            [Usvd, Ssvd, ~] = svd(X0c, 'econ');
            sv = diag(Ssvd);
            energyCum = cumsum(sv.^2) / sum(sv.^2);
            r = find(energyCum >= energyThresh, 1, 'first');
            U00 = Usvd(:, 1:r);
            clear Usvd Ssvd

            Bpre = U00' * X0c;                       % reuse X0c = Yc_b - S0
            maxColNorm = max(sqrt(sum(Bpre.^2, 1)));
            maxSigBpre = norm(Bpre);                 % top singular value
            if maxSigBpre == 0
                error('B became zero during batch 1 initialization for %s.', dsPath);
            end
            mu0 = maxColNorm / (sqrt(r/q_b) * maxSigBpre);

            U00proj = proj_mu_B(U00.', mu0, r, n, maxSigBpre).';
            [U, ~] = qr(U00proj, 0);
            T_this = T1;
            clear X0c Bpre U00 U00proj
        else
            T_this = Tb;                             % warm start
        end

        S = S0;

        gaRow = min(gamma * alpha_row, 1);
        gaCol = min(gamma * alpha_col, 1);

        if b == 1
            initTime = toc(tc);                      % batch-1 init duration (info only; still counted)
        end

        %% ---- iteration loop, rank growth, output block (timed) ----

        eta = 0;
        for t = 1:T_this
            Btilde = U' * (Yc_b - S);

            muTic = tic;
            maxColNorm = max(sqrt(sum(Btilde.^2, 1)));
            maxSigB = norm(Btilde);                  % replaces svds(.,1)
            if maxSigB == 0
                error('B became zero at iteration %d in batch %d of %s.', t, b, dsPath);
            end
            mu_t = maxColNorm / (sqrt(r/q_b) * maxSigB);
            muTimeAccum = muTimeAccum + toc(muTic);

            B = proj_mu_B(Btilde, mu_t, r, q_b, maxSigB);

            L     = U * B;                           % once per iteration
            Splus = thresh_T_alpha_vec(Yc_b - L, gaRow, gaCol);

            BBt   = B * B';
            gradU = U * BBt + (Splus - Yc_b) * B';   % == (U*B + Splus - Yc)*B'
            if t == 1
                eta = 0.5 / norm(B);                 % replaces svds(B,1)
            end
            Uplus = U - eta * gradU * pinv(BBt);
            [U, ~] = qr(Uplus, 0);

            S = Splus;
        end

        %% ---- Rank growth (same rule; Eb only built if growing) ----
        Xb = Yc_b - S;
        eb = NaN;
        rankGrew = 0;
        if b >= 2
            UtX = U' * Xb;
            nX  = norm(Xb, 'fro');
            eb  = sqrt(max(0, 1 - (norm(UtX, 'fro') / nX)^2));
            if eb > rankGrowThresh
                Eb = Xb - U * UtX;
                [u_new, ~, ~] = svds(Eb, 1);
                U = [U, u_new];
                r = r + 1;
                B = U' * Xb;
                rankGrew = 1;
            end
        end

        Xhat = U * B + mean_b;                       % background block
        computeTime = computeTime + toc(tc);
        %% ---- end timed section ----

        %% ---- Disk write + log (NOT timed) ----
        if saveAsSingle
            matObj.background(:, startIdx:endIdx) = single(Xhat);
        else
            matObj.background(:, startIdx:endIdx) = Xhat;
        end

        fprintf(logFid, '%d\t%d\t%d\t%d\t%d\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%d\n', ...
            b, startIdx, endIdx, q_b, r, ...
            mean(alpha_row), min(alpha_row), max(alpha_row), ...
            mean(alpha_col), min(alpha_col), max(alpha_col), ...
            eb, rankGrew);

        fprintf('  batch %d/%d done (q_b=%d, r=%d)\n', b, K, q_b, r);

        clear Yc_b S S0 B Splus Btilde gradU Uplus Xb Eb UtX L Xhat BBt
    end

    fclose(logFid);

    elapsedTime  = computeTime - muTimeAccum;
    timePerFrame = elapsedTime / q_total;

    %% ---- Save run-level parameters ----
    paramsPath = fullfile(seqOutDir, 'simulation_parameters.txt');
    fid = fopen(paramsPath, 'w');
    fprintf(fid, 'Dataset: %s\n', dsPath);
    fprintf(fid, 'n (pixels): %d\n', n);
    fprintf(fid, 'q_total (frames): %d\n', q_total);
    fprintf(fid, 'batchSize: %d\n', batchSize);
    fprintf(fid, 'num_batches: %d\n', K);
    fprintf(fid, 'tau_abs: %g\n', tau_abs);
    fprintf(fid, 'c_norm: %g\n', c_norm);
    fprintf(fid, 'sigma_min: %g\n', sigma_min);
    fprintf(fid, 'c_corr (alpha correction): %g\n', c_corr);
    fprintf(fid, 'gamma: %g\n', gamma);
    fprintf(fid, 'energyThresh: %g\n', energyThresh);
    fprintf(fid, 'rankGrowThresh: %g\n', rankGrowThresh);
    fprintf(fid, 'T1: %d\n', T1);
    fprintf(fid, 'Tb: %d\n', Tb);
    fprintf(fid, 'final_rank: %d\n', r);
    fprintf(fid, 'timing: all algorithm compute incl. initialization (I/O and per-iteration mu computation excluded)\n');
    fprintf(fid, 'batch1_init_time_sec (included in total_time_sec): %.4f\n', initTime);
    fprintf(fid, 'total_time_sec: %.4f\n', elapsedTime);
    fprintf(fid, 'time_per_frame_sec: %.6f\n', timePerFrame);
    fclose(fid);

    fprintf('  -> saved to %s (compute time/frame = %.6f s)\n', seqOutDir, timePerFrame);
end

fprintf('All datasets processed.\n');

%% ================= Local functions =================
function files = sort_nat_by_number(files)
    names = {files.name};
    nums = zeros(numel(names),1);
    for k = 1:numel(names)
        digs = regexp(names{k}, '\d+', 'match');
        nums(k) = str2double(digs{end});
    end
    [~, idx] = sort(nums);
    files = files(idx);
end
