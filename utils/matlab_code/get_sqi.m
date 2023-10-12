function sqi = get_sqi(args)
    mixed_signal = args.mixed_signal;
    fs = args.fs;
    model_type = args.model_type;

    if strcmp(model_type, 'heart')
        model=load('trained_heart.mat','top_features','C','S','MdlStd');
    elseif strcmp(model_type, 'lung')
        model=load('trained_lung.mat','top_features','C','S','MdlStd');
    else
        error('Invalid model_type argument')
    end
    % Heart_SQI= get_all_SQIs(mixed_signal, fs);
    Heart_SQI= get_all_SQIs_modified(mixed_signal, fs, model.top_features);

    Heart_SQI = splitvars(Heart_SQI);
    tmp1 = model.C.Properties.VariableNames;
    model.C.Properties.VariableNames = Heart_SQI.Properties.VariableNames;
    model.S.Properties.VariableNames = Heart_SQI.Properties.VariableNames;
    Heart_SQI = normalize(Heart_SQI,'center',model.C,'scale',model.S);
    Heart_SQI.Properties.VariableNames = tmp1;
    Heart_SQI = Heart_SQI (:,model.top_features);
    sqi=predict(model.MdlStd,Heart_SQI);
end
