function [alpha_row, alpha_col, mask] = estimate_alpha(Y, tau_abs, c_norm, sigma_min)
% ESTIMATE_ALPHA  Computes per-pixel alpha_row and per-frame alpha_col
% from the binary mask produced by estimate_mask.
%
%   Y         : n x q data matrix for this batch (n pixels, q frames)
%   tau_abs, c_norm, sigma_min : passed straight through to estimate_mask
%
%   alpha_row : n x 1, alpha_row(i) = sum(mask(i,:)) / q
%   alpha_col : 1 x q, alpha_col(j) = sum(mask(:,j)) / n
%   mask      : n x q binary mask (returned in case it's needed downstream)

    mask = estimate_mask(Y, tau_abs, c_norm, sigma_min);
    [n, q] = size(mask);

    alpha_row = sum(mask, 2) / q;   % n x 1, per pixel
    alpha_col = sum(mask, 1) / n;   % 1 x q, per frame
end
