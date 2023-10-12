function output = get_nmcf(mixed_signal, fs, noise)
    mixed_signal = mixed_signal';
    
    switch noise
        case 'none'
            S=load('nmf_parameters_none.mat');
        case 'Bubble'
            S=load('nmf_parameters_Bubble.mat');
        case 'CPAP'
            S=load('nmf_parameters_CPAP.mat');
        otherwise
            error('Unknown "noise" argument')
    end

    nmf_method="nmcf";
    supervision="HS LS NS NU";
    reconstruction="Filtering";

    xhat=...
        nmcf_overall2(mixed_signal,4000,nmf_method, supervision, S.TF,S.options_tf, reconstruction,S.K, S.options_nmf, S.MAXITER, S.W_1, [], S.W_3, S.W_4, S.W_5, S.W_6, S.V_h, S.V_l, S.V_n, S.V_s, S.V_r);

    nmcf_heart=xhat(1:10*fs,1);
    nmcf_lung=xhat(1:10*fs,4);

    output = [nmcf_heart, nmcf_lung]';
end