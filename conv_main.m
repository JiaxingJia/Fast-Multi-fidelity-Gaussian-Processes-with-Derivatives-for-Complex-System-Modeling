clc
clear all
close all

rng('shuffle')
addpath(genpath(fullfile(pwd)));
global GP_grad_info

exactfun = @(x,t) sin(x-5*t);
%% Experiment setting
color_style = 'jet';
load('high_fidelity_x.mat'); load('high_fidelity_t.mat'); load('high_fidelity_u.mat');
load('low_fidelity_x.mat'); load('low_fidelity_t.mat'); load('low_fidelity_u.mat');
t_max = max(high_fidelity_t); t_min = min(high_fidelity_t); x_max = max(high_fidelity_x); x_min = min(high_fidelity_x);
[T_H, X_H] = meshgrid(high_fidelity_t, high_fidelity_x);
[T_L, X_L] = meshgrid(low_fidelity_t, low_fidelity_x);
test_X = X_H; test_T = T_H; True = exactfun(test_X, test_T);
n_lx = 20; n_lt = 81; n_hx = 128; n_ht = 513; n_hs = 30; n_ls = 300; delta = 1e-5; noise_std = .02; cnum = 50; nepo = 1;
slu_l = low_fidelity_u; slu_h = high_fidelity_u;
legend_nan = {'GP','GPD','KOH','HK','FMGPD'};
%% Training model
for epo = 1:nepo
    disp(['Running group ',num2str(epo)]);
    %% Reshaping data
    True_vector = reshape(True, n_hx*n_ht, 1);
    test_x_vector = reshape(test_X, n_hx*n_ht, 1); test_t_vector = reshape(test_T, n_hx*n_ht, 1);
    slu_l_vector = reshape(slu_l, n_lx*n_lt, 1); slu_h_vector = reshape(slu_h, n_hx*n_ht, 1);
    x_l_vector = reshape(X_L, n_lx*n_lt, 1); t_l_vector = reshape(T_L, n_lx*n_lt, 1);
    x_h_vector = reshape(X_H, n_hx*n_ht, 1); t_h_vector = reshape(T_H, n_hx*n_ht, 1);
    h_term_conv(:,epo) = randperm(n_hx*n_ht)'; l_term_conv(:,epo) = randperm(n_lx*n_lt)';
    %% Training and testing dataset
    train_xl = [t_l_vector(l_term_conv(1:n_ls,epo)), x_l_vector(l_term_conv(1:n_ls,epo))]; train_yl = slu_l_vector(l_term_conv(1:n_ls,epo)) + randn(n_ls,1)*noise_std;
    train_xh = [t_h_vector(h_term_conv(1:n_hs,epo)), x_h_vector(h_term_conv(1:n_hs,epo))]; train_yh = slu_h_vector(h_term_conv(1:n_hs,epo)) + randn(n_hs,1)*noise_std;
    test_x = [test_t_vector, test_x_vector]; test_y = True_vector;
    %% GP without gradient
    [GP_grad_info.mu_gpl, ~] = gp(train_xl, train_yl); gpl_pred(:,epo) = GP_grad_info.mu_gpl(test_x);   % LF GP
    [GP_grad_info.mu_gph, ~] = gp(train_xh, train_yh); gph_pred(:,epo) = GP_grad_info.mu_gph(test_x);   % HF GP
    %% KOH
    koh_pred(:,epo) = koh(train_xl, train_yl, train_xh, train_yh, test_x);
    %% HK
    GP_grad_info.F = GP_grad_info.mu_gpl(train_xh);
    [mu_hk, ~] = gp_hk(train_xh, train_yh);
    hk_pred(:,epo) = mu_hk(test_x);
    %% Gradient
    [GX_H, GT_H] = gradient(slu_h,high_fidelity_t(1,2)-high_fidelity_t(1,1),high_fidelity_x(1,2)-high_fidelity_x(1,1));
    grad_all{1} = [reshape(GX_H,n_ht*n_hx,1),reshape(GT_H,n_ht*n_hx,1)];

    [GX_L, GT_L] = gradient(slu_l,low_fidelity_t(1,2)-low_fidelity_t(1,1),low_fidelity_x(1,2)-low_fidelity_x(1,1));
    grad_all{2} = [reshape(GX_L,n_lt*n_lx,1),reshape(GT_L,n_lt*n_lx,1)];
    %% GPDH
    [GP_grad_info.mu_gpdh, ~] = gp_grad(train_xh, train_yh, grad_all{1}(h_term_conv(1:n_hs,epo),:));
    gpdh_pred(:,epo) = GP_grad_info.mu_gpdh(test_x);
    %% FMGPD
    [GP_grad_info.mu_gpdl_SKI, ~] = gp_SKI_grad(train_xl, train_yl, grad_all{2}(l_term_conv(1:n_ls,epo),:));

    grad_all{3}(:,1) = (GP_grad_info.mu_gpdl_SKI([train_xh(:,1)+delta,train_xh(:,2)]) - GP_grad_info.mu_gpdl_SKI(train_xh))/delta;
    grad_all{3}(:,2) = (GP_grad_info.mu_gpdl_SKI([train_xh(:,1),train_xh(:,2)+delta]) - GP_grad_info.mu_gpdl_SKI(train_xh))/delta;

    GP_grad_info.F = [];   % Clear F for HK
    GP_grad_info.F = [GP_grad_info.mu_gpdl_SKI(train_xh);vec(grad_all{3})];   % LF enhanced prior for FMGPD
    [mu_fmgpd, ~] = gp_fmgpd(train_xh, train_yh, grad_all{1}(h_term_conv(1:n_hs,epo),:));
    fmgpd_pred(:,epo) = mu_fmgpd(test_x);
    %% Metrics
    % RMSE
    RMSE_GP(epo,1) = sqrt(mean((gph_pred(:,epo) - True_vector).^2));
    RMSE_GPD(epo,1) = sqrt(mean((gpdh_pred(:,epo) - True_vector).^2));
    RMSE_KOH(epo,1) = sqrt(mean((koh_pred(:,epo) - True_vector).^2));
    RMSE_HK(epo,1) = sqrt(mean((hk_pred(:,epo) - True_vector).^2));
    RMSE_FMGPD(epo,1) = sqrt(mean((fmgpd_pred(:,epo) - True_vector).^2));

    % R2
    R2_GP(epo,1) = 1 - (sum((gph_pred(:,epo) - True_vector).^2)/sum((True_vector - mean(True_vector)).^2));
    R2_GPD(epo,1) = 1 - (sum((gpdh_pred(:,epo) - True_vector).^2)/sum((True_vector - mean(True_vector)).^2));
    R2_KOH(epo,1) = 1 - (sum((koh_pred(:,epo) - True_vector).^2)/sum((True_vector - mean(True_vector)).^2));
    R2_HK(epo,1) = 1 - (sum((hk_pred(:,epo) - True_vector).^2)/sum((True_vector - mean(True_vector)).^2));
    R2_FMGPD(epo,1) = 1 - (sum((fmgpd_pred(:,epo) - True_vector).^2)/sum((True_vector - mean(True_vector)).^2));

    % MAE
    MAE_GP(epo,1) = max(abs(True_vector - gph_pred(:,epo)));
    MAE_GPD(epo,1) = max(abs(True_vector - gpdh_pred(:,epo)));
    MAE_KOH(epo,1) = max(abs(True_vector - koh_pred(:,epo)));
    MAE_HK(epo,1) = max(abs(True_vector - hk_pred(:,epo)));
    MAE_FMGPD(epo,1) = max(abs(True_vector - fmgpd_pred(:,epo)));
end
%% Celling the Results
results_cell = {gph_pred, gpdh_pred, koh_pred, hk_pred, fmgpd_pred};
results_cell_median = {median(gph_pred,2), median(gpdh_pred,2), median(koh_pred,2), median(hk_pred,2), median(fmgpd_pred,2)};
%% Plot Average Results
maxval_pred = max(cellfun(@max, results_cell_median)); minval_pred = min(cellfun(@min, results_cell_median));

subplot(2,3,1)
colormap(color_style);
contourf(test_T, test_X, True, cnum, 'LineWidth', .6, 'LineColor','none'); hold on;
xlabel('t'); ylabel('x'); title('Reference')
clim([minval_pred,maxval_pred]);

for i=1:length(results_cell)
    subplot(2,3,i+1)
    colormap(color_style);
    contourf(test_T, test_X, reshape(median(results_cell{i},2), n_hx, n_ht), cnum, 'LineWidth', .6, 'LineColor','none'); hold on
    title(legend_nan{i});
    xlabel('t'); ylabel('x');
    clim([minval_pred,maxval_pred]);
end
%% Plot Absolute Error
maxval_error = -Inf;
for i=1:length(results_cell)
    abs_error{i} = abs(True-reshape(median(results_cell{i},2), n_hx, n_ht));
    maxval_error = max(maxval_error, max(abs_error{i}(:)));
end
figure;
for i=1:length(results_cell)
    subplot(2,3,i)
    colormap(color_style);
    surf(test_T, test_X, abs(True-reshape(median(results_cell{i},2), n_hx, n_ht)), ...
        'EdgeColor','none','LineStyle','none','FaceLighting','gouraud');
    title(legend_nan{i});
    xlabel('t'); ylabel('x'); zlabel('Absolute error', 'Interpreter', 'latex');
    xlim([t_min,t_max]); ylim([x_min,x_max]);
    set(get(gca,'XLabel'),'Fontname','Times', 'Fontsize', 11);
    set(get(gca,'YLabel'),'Fontname','Times', 'Fontsize', 11);
    set(get(gca,'ZLabel'),'Fontname','Times', 'Fontsize', 11);
    zlim([0,maxval_error]);
    clim([0,maxval_error]);
end