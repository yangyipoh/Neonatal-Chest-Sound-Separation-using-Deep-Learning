function pretrain_nmf(respiratory_support, base_path, fold)
    % expect args.resp_support, args.base_path, args.fold
    % respiratory_support = 'Bubble, 'CPAP', 'none'
    options_tf.FFTSIZE = 1024;
    options_tf.HOPSIZE = 256;
    options_tf.WINDOWSIZE =512;
    max_examples=10;

    heart_path= fullfile(base_path, sprintf('TrainHeart%d', fold));
    lung_path= fullfile(base_path, sprintf('TrainLung%d', fold));
    cry_path= fullfile(base_path, sprintf('TrainCry%d', fold));
    stmv_path= fullfile(base_path, sprintf('TrainStmv%d', fold));
    bubble_path= fullfile(base_path, sprintf('TrainBubble%d', fold));
    cpap_path= fullfile(base_path, sprintf('TrainCPAP%d', fold));
    
    TF='STFT';
    
    options_nmf.W1=0;
    options_nmf.W3=1;
    options_nmf.W4=0;
    options_nmf.W5=0.25;
    options_nmf.W6=0.25;
    
    options_nmf.beta_loss=1;
    options_nmf.sparsity=0.1;
    MAXITER = 100;
    K=[20 10 20 20 20 20];
    
    [V_h, V_l, V_n,V_s, V_r, W_1, W_3, W_4, W_5, W_6]=...
        load_example2(respiratory_support, TF,options_tf, max_examples,heart_path,lung_path,cry_path,stmv_path,bubble_path,cpap_path,options_nmf,MAXITER,K);
    save(sprintf('heart_sound_analysis/nmf_parameters_%s.mat', respiratory_support), 'TF', "options_tf", "K", "options_nmf", "MAXITER", ...
        "W_1", "W_3", "W_4", "W_5", "W_6", "V_h", "V_l", "V_n", "V_s", "V_r")
end