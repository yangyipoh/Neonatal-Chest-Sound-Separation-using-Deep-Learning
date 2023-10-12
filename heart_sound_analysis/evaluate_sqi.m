clear variables; close all; clc;

addpath(genpath('Breathing Rate and Segmentation'))
addpath(genpath('Cardiac_Murmur'))
addpath(genpath('Heart Rate and Segmentation'))
addpath(genpath('Signal Quality'))
addpath(genpath('Sound Separation'))
addpath(genpath('yamnet'))

EVAL_DIR = '/srv/data/Data/Sync Dataset/Data/Audio Files/Sync Segmented (Ethan)';
FOLDERS = {'Bubble', 'CPAP', 'Nil', 'Ventilator'};
EVAL_SHEET = fullfile(EVAL_DIR, 'data.csv');
EVAL_FILE = '~/Documents/Code/bss-conformer/result/sqi.csv';;
METHOD = 'NMCF';

% read excel sheet
T_out = table();
count = 1;

% load NMCF parameters
if strcmp(METHOD, 'NMCF')
    args_bubble = load_NMCF_param('Bubble');
    args_cpap = load_NMCF_param('CPAP');
    args_nil = load_NMCF_param('NoNoise');
end

for idx1 = 1:4
    my_path = dir(fullfile(EVAL_DIR, FOLDERS{idx1}));
    files = {my_path.name};
    files = files(3:end);
    N = length(files);
    for idx2 = 1:N
        % read audio file
        my_path = fullfile(EVAL_DIR, FOLDERS{idx1}, files{idx2});
        [waveform, ~] = audioread(my_path);

        % pass it through the model
        if strcmp(METHOD, 'NMF')
            args.fs = 4000;
            args.mixed_signal = waveform;
            est = get_nmf(args);
        else
            if strcmp(FOLDERS{idx1}, 'Bubble')
                args_bubble.mixed_signal = waveform;
                args_bubble.fs = 4000;
                est = get_nmcf(args_bubble);
            elseif strcmp(FOLDERS{idx1}, 'CPAP')
                args_cpap.mixed_signal = waveform;
                args_cpap.fs = 4000;
                est = get_nmcf(args_cpap);
            else
                args_nil.mixed_signal = waveform;
                args_nil.fs = 4000;
                est = get_nmcf(args_nil);
            end
        end
        heart_signal = est(:, 1);
        lung_signal = est(:, 2);

        % calculate SQI
        % if idx2 == 57 && idx1 == 3  % skip this for NMCF because it gets stuck in an infinite loop trying to get some metric that doesn't work
        %     sqi_h = 0;
        %     sqi_l = 0;
        % else
            [sqi_h, sqi_l] = eval_sqi(heart_signal, lung_signal);
        % end

        % append metric
        T_out.Filename(count) = files(idx2);
        T_out.PredictHeart(count) = sqi_h;
        T_out.PredictLung(count) = sqi_l;
        count = count + 1;
        fprintf('Completed %d of %d of Folder %d\n', idx2, N, idx1);
    end
end
writetable(T_out, EVAL_FILE);
fprintf('Evaluation Complete');

function [sqi_h, sqi_l] = eval_sqi(heart_signal, lung_signal)
    args.fs = 4000;
    args.model_type = 'heart';
    args.mixed_signal = heart_signal;
    sqi_h = get_sqi(args);
    args.model_type = 'lung';
    args.mixed_signal = lung_signal;
    sqi_l = get_sqi(args);
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