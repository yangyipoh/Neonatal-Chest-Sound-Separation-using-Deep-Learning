function HR_est = get_hr(heart_signal, fs)
    heart_signal = heart_signal';
    % Parameters
    max_HR=200;
    min_HR=110;
    options_hr.env='hilbert';
    options_hr.autocorr='filtered';
    options_hr.init_hr='autocorr_peak';
    options_hr.systolic='yes';
    options_hr.seg='springer';

    load('Springer_B_matrix.mat', 'Springer_B_matrix');
    load('Springer_pi_vector.mat', 'Springer_pi_vector');
    load('Springer_total_obs_distribution.mat', 'Springer_total_obs_distribution');

    options_hr.seg_fs=50;
    options_hr.seg_pi_vector=Springer_pi_vector;
    options_hr.seg_b_matrix=Springer_B_matrix;
    options_hr.seg_total_obs_dist=Springer_total_obs_distribution;

    HR = get_hr_segmentation(heart_signal, fs, max_HR,min_HR,options_hr);

    HR_est=HR.seg_hr;
end