clear all; close all; clc;

addpath(genpath('heart_sound_analysis'))
addpath('utils/matlab_code')

% batch = 16;
total_runs = 10;
total_time = 0;
for idx = 1:total_runs
    input_wav = randn(1, 40000);  % replace this by loading some sample

    f = @() get_nmcf(input_wav, 4000, 'none');
    time_taken = timeit(f);
    total_time = total_time + time_taken;
end

time_per_epoch = (total_time/total_runs);
fprintf('Time completed: %.4f\n', time_per_epoch)