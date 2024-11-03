% Interface for training a GP with gradients using the SE Kernel
% 
% [mu, K] = gp_grad(X, Y, DY)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: dense kernel matrix

function [mu, K] = gp_fmgpd(X, Y, DY)
global GP_grad_info
[ntrain, d] = size(X);
beta = 1e-5;

% Train FMGPD
cov = @(hyp) se_kernel_grad(X, hyp);
lmlfun = @(x) lmlh_exact(cov, [Y, DY], x, beta);

hyp = GP_grad_info.lf_params;
params = minimize(hyp, lmlfun, -100);

sigma = sqrt(exp(2*params.lik) + beta);
fprintf('SE with gradients: (ell, s, sigma1, sigma2) = (%.3f, %.3f, %.3f, %.3f)\n', exp(params.cov), sigma)

% Calculate interpolation coefficients
sigma2 = [sigma(1)*ones(1, ntrain), sigma(2)*ones(1, ntrain*d)].^2;
K = se_kernel_grad(X, params) + diag(sigma2);
lambda = K\(vec([Y, DY]) - GP_grad_info.belta0 * GP_grad_info.F);

% Function handle returning GP mean to be output
mu = @(XX) mean_grad(XX, X, lambda, params);
end

function ypred = mean_grad(XX, X, lambda, params)
global GP_grad_info
KK = se_kernel_grad(X, params, XX);
ypred = GP_grad_info.belta0 * GP_grad_info.mu_gpdl_SKI(XX) + KK*lambda;
end