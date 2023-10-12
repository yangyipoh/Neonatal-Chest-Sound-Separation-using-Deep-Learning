clear variables; close all; clc;

% execute warning('off') to suppress messages

addpath('bss')
addpath(genpath('Breathing Rate and Segmentation'))
addpath(genpath('Cardiac_Murmur'))
addpath(genpath('Heart Rate and Segmentation'))
addpath(genpath('Signal Quality'))
addpath(genpath('Sound Separation'))
addpath(genpath('yamnet'))


EVAL_DIR = '/srv/data/Data/Evaluation Set 2';
EVAL_SHEET = fullfile(EVAL_DIR, 'Evaluation Set.xlsx');
NOISE_TYPES = {'Cry', 'NoNoise', 'Stmv'};

for idx_noise = 1:3
    NOISE_TYPE = NOISE_TYPES{idx_noise};
    METHOD = 'NMCF';
    EVAL_FILE = fullfile('~/Documents/Code/bss-conformer/result', sprintf('%s.csv', NOISE_TYPE));

    % read excel sheet
    T = readtable(EVAL_SHEET, 'Sheet', NOISE_TYPE);
    T_out = table();
    count = 1;

    % load NMCF parameters if needed
    if strcmp(METHOD, 'NMCF')
        args = load_NMCF_param(NOISE_TYPE);
    end

    for idx = 1:height(T)
        % skip values if its not needed
        if ~strcmp(NOISE_TYPE, 'NoNoise') && T.Include_(idx) == 0
            continue
        end

        % get signal information
        fold_idx = T.x_Fold_(idx);
        mix_idx = T.x_File_(idx);
        heart_idx = str2double(strip(T.x_Heart_(idx), 'both', "'"));
        lung_idx = str2double(strip(T.x_Lung_(idx), 'both', "'"));

        % read audio file
        my_path = fullfile(EVAL_DIR, sprintf('%s%d', NOISE_TYPE, fold_idx), sprintf('%d.wav', mix_idx));
        if ~isfile(my_path)
            continue
        end
        [mix_wav, ~] = audioread(my_path);
        my_path = fullfile(EVAL_DIR, sprintf('TestHeart%d', fold_idx), sprintf('%d.wav', heart_idx));
        if ~isfile(my_path)
            continue
        end
        [test_heart_wav, ~] = audioread(my_path);
        my_path = fullfile(EVAL_DIR, sprintf('TestLung%d', fold_idx), sprintf('%d.wav', lung_idx));
        if ~isfile(my_path)
            continue
        end
        [test_lung_wav, ~] = audioread(my_path);

        % base case
        est = [mix_wav, mix_wav];
        ref = [test_heart_wav, test_lung_wav];
        [sdr_h_m, sdr_l_m, sir_h_m, sir_l_m, sisdr_h_m, sisdr_l_m] = evaluate_sxr(est', ref');

        % evaluate model with SDR
        args.fs = 4000;
        args.mixed_signal = mix_wav;
        if strcmp(METHOD, 'NMF')
            est = get_nmf(args);
        else
            est = get_nmcf(args);
        end
        [sdr_h_f, sdr_l_f, sir_h_f, sir_l_f, sisdr_h_f, sisdr_l_f] = evaluate_sxr(est', ref');
        
        % calculate performance improvement
        T_out.sdr_h_mix(count) = sdr_h_m;
        T_out.sdr_l_mix(count) = sdr_l_m;
        T_out.sir_h_mix(count) = sir_h_m;
        T_out.sir_l_mix(count) = sir_l_m;
        T_out.sisdr_h_mix(count) = sisdr_h_m;
        T_out.sisdr_l_mix(count) = sisdr_l_m;

        T_out.sdr_h_filter(count) = sdr_h_f;
        T_out.sdr_l_filter(count) = sdr_l_f;
        T_out.sir_h_filter(count) = sir_h_f;
        T_out.sir_l_filter(count) = sir_l_f;
        T_out.sisdr_h_filter(count) = sisdr_h_f;
        T_out.sisdr_l_filter(count) = sisdr_l_f;

        T_out.sdr_h_improve(count) = sdr_h_f - sdr_h_m;
        T_out.sdr_l_improve(count) = sdr_l_f - sdr_l_m;
        T_out.sir_h_improve(count) = sir_h_f - sir_h_m;
        T_out.sir_l_improve(count) = sir_l_f - sir_l_m;
        T_out.sisdr_h_improve(count) = sisdr_h_f - sisdr_h_m;
        T_out.sisdr_l_improve(count) = sisdr_l_f - sisdr_l_m;

        count = count + 1;

        fprintf('Completed %d of %d\n', idx, height(T));
    end
    writetable(T_out, EVAL_FILE);
    fprintf('Evaluation Complete');
end

function [sdr_h, sdr_l, sir_h, sir_l, sisdr_h, sisdr_l] = evaluate_sxr(est, ref)
    % est -> (2,L), ref -> (2,L)
    [sdr, sir, ~, ~] = bss_eval_sources(est, ref);
    sdr_h = sdr(1, 1);
    sdr_l = sdr(2, 1);
    sir_h = sir(1, 1);
    sir_l = sir(2, 1);
    
    % sisdr stuff
    [s_target, e_inter, e_artif] = bss_decomp_gain(est(1, :), 1, ref);
    [sisdr_h, ~, ~] = bss_crit(s_target, e_inter, e_artif);
    [s_target, e_inter, e_artif] = bss_decomp_gain(est(2, :), 2, ref);
    [sisdr_l, ~, ~] = bss_crit(s_target, e_inter, e_artif);
end

function args = load_NMCF_param(noise_type)
    switch noise_type
        case 'Bubble'
            respiratory_support = 'Bubble';
        case 'CPAP'
            respiratory_support = 'CPAP';
        case 'Cry'
            respiratory_support = 'none';
        case 'NoNoise'
            respiratory_support = 'none';
        case 'Stmv'
            respiratory_support = 'none';
    end
    TF='STFT';
    options_tf.FFTSIZE = 1024;
    options_tf.HOPSIZE = 256;
    options_tf.WINDOWSIZE =512;
    
    options_nmf.W1=0; % heart
    options_nmf.W3=1; % noise 1
    options_nmf.W4=0; % lung 
    options_nmf.W5=0.25; % noise 2
    options_nmf.W6=0.25; % noise 3
    options_nmf.beta_loss=1;
    options_nmf.sparsity=0.1;
    MAXITER = 100;
    K=[20 10 20 20 20 20];
    
    max_examples=10;
    heart_path= '/home/yypoh/Documents/Code/bss-conformer/heart_sound_analysis/Sound Separation/Reference Sounds/Heart';
    lung_path= '/home/yypoh/Documents/Code/bss-conformer/heart_sound_analysis/Sound Separation/Reference Sounds/Lung';
    cry_path= '/home/yypoh/Documents/Code/bss-conformer/heart_sound_analysis/Sound Separation/Reference Sounds/Cry';
    stmv_path= '/home/yypoh/Documents/Code/bss-conformer/heart_sound_analysis/Sound Separation/Reference Sounds/Stmv';
    bubble_path= '/home/yypoh/Documents/Code/bss-conformer/heart_sound_analysis/Sound Separation/Reference Sounds/Bubble';
    cpap_path= '/home/yypoh/Documents/Code/bss-conformer/heart_sound_analysis/Sound Separation/Reference Sounds/CPAP';
    [V_h, V_l, V_n,V_s, V_r, W_1, W_3, W_4, W_5, W_6]=...
        load_example2(respiratory_support, TF,options_tf, max_examples,heart_path,lung_path,cry_path,stmv_path,bubble_path,cpap_path,options_nmf,MAXITER,K);
    args.V_h = V_h;
    args.V_l = V_l;
    args.V_n = V_n;
    args.V_s = V_s;
    args.V_r = V_r;
    args.W_1 = W_1;
    args.W_3 = W_3;
    args.W_4 = W_4;
    args.W_5 = W_5;
    args.W_6 = W_6;
end