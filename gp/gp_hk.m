% Interface for training a GP using the SE Kernel
% 
% [mu, K] = gp(X, Y)
% 
% Input
% X: n by d matrix representing n training points in d dimensions
% Y: training values corresponding Y = f(X)
% Output
% mu: mean function handle such that calling mu(XX) for some predictive points XX calculates the mean of the GP at XX 
% K: dense kernel matrix

function [mu, K] = gp_hk(X, Y)
global GP_grad_info
[ntrain, d] = size(X);

% Initial hyperparameters
ell0 = 2;
s0 = std(Y);
sig0 = 0.5; 
beta = 1e-6;

% Train GP 
cov = @(hyp) se_kernel(X, hyp);
lmlfun = @(x) lmlh_exact(cov,Y, x, beta);
hyp = struct('cov', log([ell0, s0]), 'lik', log([sig0]));
params = minimize_quiet(hyp, lmlfun, -100);
sigma = sqrt(exp(2*params.lik) + beta);
fprintf('SE with gradients: (ell, s, sigma1) = (%.3f, %.3f, %.3f)\n', exp(params.cov), sigma)

% Calculate interpolation coefficients
sigma2 = sigma^2*ones(1, ntrain);
K = se_kernel(X, params) + diag(sigma2);
lambda = K\(vec(Y) - GP_grad_info.F*GP_grad_info.belta0);

% Function handle returning GP mean to be output
mu = @(XX) mean(XX, X, lambda, params);
end

function ypred = mean(XX, X, lambda, params)
global GP_grad_info
KK = se_kernel(X, params, XX);
ypred = GP_grad_info.mu_gpl(XX) * GP_grad_info.belta0 + KK*lambda;
end