function DeletedFeatureID = W_Calculate_ForRFE_DeletedID_leaveoneout(Subjects_Data, Subjects_Label, EliminationQuantity, ith, Pre_Method)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% EliminationQuantity:
%           features to eliminate one time during SVM-RFE
%
% Subject_Label:
%           array of 1 or -1
%
% ResultantFolder:
%           the path of folder storing resultant files
%

Subjects_Quantity = length(Subjects_Label);

for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    Subjects_Data_tmp = Subjects_Data;
    Subjects_Label_tmp = Subjects_Label;
    % Select training data and testing data
    test_label = Subjects_Label_tmp(i);
    test_data = Subjects_Data_tmp(i, :);
    
    Subjects_Label_tmp(i) = [];
    Subjects_Data_tmp(i, :) = [];
    Training_group1_Index = find(Subjects_Label_tmp == 1);
    Training_group0_Index = find(Subjects_Label_tmp == -1);
    Training_group1_data = Subjects_Data_tmp(Training_group1_Index, :);
    Training_group0_data = Subjects_Data_tmp(Training_group0_Index, :);
    Training_group1_Label = Subjects_Label_tmp(Training_group1_Index);
    Training_group0_Label = Subjects_Label_tmp(Training_group0_Index);
    
    %Normalizing
    Training_all_data = [Training_group1_data; Training_group0_data];
    Label = [Training_group1_Label Training_group0_Label];

    if strcmp(Pre_Method, 'Normalize')
        %Normalizing
        MeanValue = mean(Training_all_data);
        StandardDeviation = sqrt(var(Training_all_data));
        [rows, columns_quantity] = size(Training_all_data);
        for j = 1:columns_quantity
            Training_all_data(:, j) = (Training_all_data(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        [rows, columns_quantity] = size(Training_all_data);
        MinValue = min(Training_all_data);
        MaxValue = max(Training_all_data);
        if MaxValue ~= 0
            for j = 1:columns_quantity
                Training_all_data(:, j) = (Training_all_data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
            end
        end
    end

    % SVM classification
    Label = reshape(Label, length(Label), 1);
    Training_all_data = double(Training_all_data);
    model(i) = svmtrain(Label, Training_all_data,'-t 0');

end

% Averaging w
w_average = 0;
for i = 1:Subjects_Quantity
    clear w;
    w = zeros(size(model(i).SVs(1, :)));
    for j = 1 : model(i).nSV(1)
        w = w + model(i).sv_coef(j) * model(i).SVs(j, :);
    end
    for j = model(i).nSV(1) + 1 : model(i).nSV(1) + model(i).nSV(2)
        w = w + model(i).sv_coef(j) * model(i).SVs(j, :);
    end
    w_average = w_average + w;
end
w_average = w_average / Subjects_Quantity;
w_Brain = w_average;

w_Brain = w_Brain .^ 2;
[RankingResults OriginPos] = sort(w_Brain); % Default is 'ascend'

if length(OriginPos) >= ith * EliminationQuantity
    DeletedFeatureID = OriginPos([ith * EliminationQuantity : -1 : (ith - 1) * EliminationQuantity + 1]);
else
    DeletedFeatureID = OriginPos([end : -1 : (ith - 1) * EliminationQuantity + 1]);
end

