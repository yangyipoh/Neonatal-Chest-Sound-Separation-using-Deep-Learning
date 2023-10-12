clear variables; close all; clc;

addpath(genpath('Breathing Rate and Segmentation'))
addpath(genpath('Cardiac_Murmur'))
addpath(genpath('Heart Rate and Segmentation'))
addpath(genpath('Signal Quality'))
addpath(genpath('Sound Separation'))
addpath(genpath('yamnet'))

EVAL_DIR = '/srv/data/Data/Sync Dataset/Data/Audio Files/Sync Segmented';
EVAL_SHEET = fullfile(EVAL_DIR, 'data.csv');
METHOD = 'NMCF';
EVAL_FILE = '~/Documents/Code/bss-conformer/result/hrbr.csv';

% read excel sheet
T = readtable(EVAL_SHEET);
T_out = table();
count = 1;

% load NMCF parameters
if strcmp(METHOD, 'NMCF')
    args_noise = load_NMCF_param('Bubble');
    args_nil = load_NMCF_param('NoNoise');
end

for idx = 1:height(T)
    % read audio file
    my_path = fullfile(EVAL_DIR, T.Filename{idx});
    [waveform, ~] = audioread(my_path);

    % pass waveform to model
    if strcmp(METHOD, 'NMF')
        args.fs = 4000;
        args.mixed_signal = waveform;
        est = get_nmf(args);
    else
        if T.RespSupport(idx) == 1
            args_noise.mixed_signal = waveform;
            args_noise.fs = 4000;
            est = get_nmcf(args_noise);
        else
            args_nil.mixed_signal = waveform;
            args_nil.fs = 4000;
            est = get_nmcf(args_nil);
        end
    end
    heart_signal = est(:, 1);
    lung_signal = est(:, 2);

    % predicted HR and BR
    [hr, br] = eval_hr(heart_signal, lung_signal);
    
    % append
    T_out.Filename(count) = T.Filename(idx);
    T_out.PredictHR(count) = hr;
    T_out.PredictBR(count) = br;
    count = count + 1;

    fprintf('Completed %d of %d\n', idx, height(T));
end
writetable(T_out, EVAL_FILE);
fprintf('Evaluation Complete\n');

function [hr, br] = eval_hr(heart_signal, lung_signal)
    args.heart_signal = heart_signal;
    args.lung_signal = lung_signal;
    args.fs = 4000;

    hr = get_hr(args);
    br = get_br(args);
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