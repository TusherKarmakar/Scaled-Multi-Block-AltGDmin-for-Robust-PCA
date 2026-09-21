function S = thresh_T_alpha_vec(M, alpha_row, alpha_col)
% THRESH_T_ALPHA_VEC  Row-AND-column thresholding where BOTH alpha_row
% and alpha_col are vectors (not scalars) - each row and each column
% gets its own independent budget.
%
%   M         : n x q matrix (residual, e.g. Y - U*B)
%   alpha_row : n x 1 (or 1 x n) vector, one alpha per pixel/row i
%   alpha_col : 1 x q (or q x 1) vector, one alpha per frame/column j
%
% Keeps M(i,j) only if it is simultaneously among the top
% ceil(alpha_row(i)*q) entries of row i AND among the top
% ceil(alpha_col(j)*n) entries of column j.

    [n, q] = size(M);
    alpha_row = alpha_row(:);        % force n x 1
    alpha_col = alpha_col(:)';       % force 1 x q

    if numel(alpha_row) ~= n
        error('alpha_row must have length n (one value per row of M).');
    end
    if numel(alpha_col) ~= q
        error('alpha_col must have length q (one value per column of M).');
    end

    absM = abs(M);

    kRow = max(1, ceil(alpha_row * q));   % n x 1, one k per row
    kCol = max(1, ceil(alpha_col * n));   % 1 x q, one k per column

    % --- per-row threshold: kRow(i)-th largest value in row i ---
    sortedRows = sort(absM, 2, 'descend');            % n x q
    rowIdx = sub2ind(size(sortedRows), (1:n)', kRow);
    rowThreshold = sortedRows(rowIdx);                % n x 1

    % --- per-column threshold: kCol(j)-th largest value in column j ---
    sortedCols = sort(absM, 1, 'descend');            % n x q
    colIdx = sub2ind(size(sortedCols), kCol, 1:q);
    colThreshold = sortedCols(colIdx);                % 1 x q

    keepMask = (absM >= rowThreshold) & (absM >= colThreshold);
    S = M .* keepMask;
end
