function output = get_traditional_method(mixed, method)
    mixed = mixed';

    switch method
        case 'ale'
            mu=0.000001;
            L=256;
            Delta = 10;
            iterations=50;
            adapt_method='Normalized LMS';
            [heart,lung]=adaptive_line_enhancement(mixed,mu,L,Delta,iterations,adapt_method);
        case 'rls'
            [heart,lung]=rls_filter(mixed,4000);
        case 'wtst'
            [heart,lung]=WTST_NST_filter(mixed,4000,3.5);
        case 'fi'
            [heart,lung]=frequency_interpolation(mixed,4000,'springer','all','linear');
        case 'mf'
            [heart,lung]=modulation_filtering(mixed,4000,[4 20]);
        case 'ssa'
            [heart,lung]=singular_spectrum_analysis(mixed,4000);
        case 'sf'
            [heart,lung]=recursive_filter(mixed,4000);
        case 'aft'
            [heart,lung]=adf_filtering(mixed,4000,'springer',5);
        case 'emd'
            [heart,lung]=emd_separation(mixed,4000,'eemd','log energy');
        case 'wssa'
            heart=wavelet_ssa(mixed,4000);
            lung = zeros(size(heart));      % lung sound is not supported
        case 'nmfc1'
            xhat=nmf_cluster1(mixed,4000);
            heart=xhat(:,1);
            lung=xhat(:,2);
        case 'nmfc2'
            % set base path from NMF Clustering 2
            base_path = 'C:\Users\ypoh0004\Documents\Data\Evaluation Set 2';
            fold = 1;

            heart_path=fullfile(base_path, sprintf('TrainHeart%d', fold));
            [Wh]=nmf_cluster2_training(heart_path);
            xhat= nmf_cluster2(mixed,4000,Wh);
            heart=resample(xhat(:,1),4000,8000);
            lung=resample(xhat(:,2),4000,8000);
        otherwise
            error('Unknown noise sound')
    end

    output = [heart, lung]';
end