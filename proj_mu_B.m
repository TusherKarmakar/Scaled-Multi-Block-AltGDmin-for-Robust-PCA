function Mout = proj_mu_B(M, mu, r, q, largest_sv)
% PROJ_MU_B  Column-wise incoherence projection.
% Clips each column of M so its 2-norm doesn't exceed
% bound = mu * sqrt(r/q) * largest_sv. Columns already under
% the bound are left untouched (min(1, bound/nk) never scales up).
%
%   M          : matrix whose columns are projected
%   mu         : incoherence parameter
%   r          : current rank
%   q          : number of columns (frames in this batch)
%   largest_sv : sigma_1, the current largest singular value

    bound = mu * sqrt(r / q) * largest_sv;
    Mout = M;
    for k = 1:q
        nk = norm(M(:, k), 2);
        Mout(:, k) = M(:, k) * min(1, (bound / nk));
    end
end
