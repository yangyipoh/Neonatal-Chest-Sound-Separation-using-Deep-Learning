function BR_est = get_br(lung_signal, fs)
    lung_signal = lung_signal';
    % Lung Parameters Lung Power
    min_BR=15;
    max_BR=100;
    options_br.window_length_br=10;
    options_br.env='psd1';
    options_br.autocorr='filtered';
    options_br.init_hr= 'envelope_findpeaks_br';
    options_br.seg='none';
    
    BR= get_br_segmentation(lung_signal, fs, max_BR,min_BR,options_br);
    BR_est=BR.initial_2;
end