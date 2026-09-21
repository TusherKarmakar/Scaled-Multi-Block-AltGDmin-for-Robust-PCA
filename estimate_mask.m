function mask = estimate_mask(Y, tau_abs, c_norm, sigma_min)
[~,q] = size(Y);
B = median(Y,2);
R = Y - B;
g = median(R,1);
R = R - g;
rMedian = median(R,2);
Rc = R - rMedian;
sigma = 1.4826 .* median(abs(Rc),2);
sigmaEff = max(sigma,sigma_min);
A = abs(Rc);
Z = A ./ sigmaEff;
mask0 = (A > tau_abs) & (Z > c_norm);
mask = false(size(mask0));
if q >= 3
    tempCount = ...
        uint8(mask0(:,1:q-2)) + ...
        uint8(mask0(:,2:q-1)) + ...
        uint8(mask0(:,3:q));
    mask(:,2:q-1) = tempCount >= 2;
    mask(:,1) = mask0(:,1) & mask0(:,2);
    mask(:,q) = mask0(:,q-1) & mask0(:,q);
else
    mask = mask0;
end
end