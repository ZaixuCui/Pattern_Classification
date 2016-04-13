function Accuracy = Fisher_Logistic_2group(Subjects_Data, Subjects_Label, alpha, Pre_Method, Cross_Validation, type, ResultantFolder, Proportion)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 0 or 1
%
% Pre_Method:
%           'Normalize' or 'Scale'
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[Subjects_Quantity Feature_Quantity] = size(Subjects_Data);

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
    Training_group0_Index = find(Subjects_Label_tmp == 0);
    Training_group1_data = Subjects_Data_tmp(Training_group1_Index, :);
    Training_group0_data = Subjects_Data_tmp(Training_group0_Index, :);
    Training_group1_Label = Subjects_Label_tmp(Training_group1_Index);
    Training_group0_Label = Subjects_Label_tmp(Training_group0_Index);
    
    All_Training = [Training_group1_data; Training_group0_data];
    Label = [Training_group1_Label Training_group0_Label];
    
    % Fisher ranking
    [tmp FeatureQuantity] = size(All_Training);
    
    if strcmp(Cross_Validation, 'normal')
        if strcmp(type, 'Threshold')
            [tmp1 RetainID tmp2] = Ranking_Fisher(All_Training, Label, 'normal', round(FeatureQuantity*Proportion));
            All_Training_New = All_Training(:, RetainID);
        elseif strcmp(type, 'BiggerThanMean')
            [tmp1 tmp2 RetainID_BiggerThanMean] = Ranking_Fisher(All_Training, Label, 'normal');
            All_Training_New = All_Training(:, RetainID_BiggerThanMean);
        end
    elseif strcmp(Cross_Validation, 'leave-one-out')
        if strcmp(type, 'Threshold')
            [tmp1 RetainID tmp2] = Ranking_Fisher(All_Training, Label, 'leave-one-out', round(FeatureQuantity*Proportion));
            All_Training_New = All_Training(:, RetainID);
        elseif strcmp(type, 'BiggerThanMean')
            [tmp1 tmp2 RetainID_BiggerThanMean] = Ranking_Fisher(All_Training, Label, 'leave-one-out');
            All_Training_New = All_Training(:, RetainID_BiggerThanMean);
        end
    end

    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        MeanValue = mean(All_Training_New);
        StandardDeviation = std(All_Training_New);
        [rows, columns_quantity] = size(All_Training_New);
        for j = 1:columns_quantity
            if StandardDeviation(j)
                All_Training_New(:, j) = (All_Training_New(:, j) - MeanValue(j)) / StandardDeviation(j);
            end
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(All_Training_New);
        MaxValue = max(All_Training_New);
        [rows, columns_quantity] = size(All_Training_New);
        for j = 1:columns_quantity
            All_Training_New(:, j) = (All_Training_New(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end

    % Logistic classification
    Label = reshape(Label, length(Label), 1);
    All_Training_New = double(All_Training_New);
    theta = Logistic_train(All_Training_New, Label, alpha);

    if strcmp(type, 'Threshold')
        test_data_New = test_data(RetainID);
    else
        test_data_New = test_data(RetainID_BiggerThanMean);
    end
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data_New = (test_data_New - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data_New = (test_data_New - MinValue) ./ (MaxValue - MinValue);
    end

    % predicts
    test_data_New = double(test_data_New);
    [predicted_labels(i), decision_values(i)] = Logistic_predict(theta, test_data_New, test_label);
    
end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == 0);
Category_group1 = predicted_labels(Group1_Index);
Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);
Y_group0 = decision_values(Group0_Index);

save([ResultantFolder filesep 'Y.mat'], 'Y_group1', 'Y_group0');
save([ResultantFolder filesep 'Category.mat'], 'Category_group1', 'Category_group0');

group0_Wrong_ID = find(Category_group0 == 1);
group0_Wrong_Quantity = length(group0_Wrong_ID);
group1_Wrong_ID = find(Category_group1 == 0);
group1_Wrong_Quantity = length(group1_Wrong_ID);
disp(['group0: ' num2str(group0_Wrong_Quantity) ' subjects are wrong ' mat2str(group0_Wrong_ID) ]);
disp(['group1: ' num2str(group1_Wrong_Quantity) ' subjects are wrong ' mat2str(group1_Wrong_ID) ]);
save([ResultantFolder filesep 'WrongInfo.mat'], 'group0_Wrong_Quantity', 'group0_Wrong_ID', 'group1_Wrong_Quantity', 'group1_Wrong_ID');
Accuracy = (Subjects_Quantity - group0_Wrong_Quantity - group1_Wrong_Quantity) / Subjects_Quantity;
disp(['Accuracy is ' num2str(Accuracy) ' !']);
save([ResultantFolder filesep 'Accuracy.mat'], 'Accuracy');

% Calculating w
W_Calculate_Logistic(Subjects_Data, Subjects_Label, alpha, Pre_Method, ResultantFolder);
