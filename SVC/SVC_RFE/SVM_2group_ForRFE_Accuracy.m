function [Accuracy w_Brain]= SVM_2group_ForRFE_Accuracy(Subjects_Data, Subjects_Label, Pre_Method)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
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
        [~, columns_quantity] = size(Training_all_data);
        for j = 1:columns_quantity
            Training_all_data(:, j) = (Training_all_data(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(Training_all_data);
        MaxValue = max(Training_all_data);
        [~, columns_quantity] = size(Training_all_data);
        for j = 1:columns_quantity
            Training_all_data(:, j) = (Training_all_data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end

    % SVM classification
    Label = reshape(Label, length(Label), 1);
    Training_all_data = double(Training_all_data);
    model(i) = svmtrain(Label, Training_all_data, '-t 0');
    
    % NC_female is 1, Nonmosaic is 0
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data = (test_data - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data = (test_data - MinValue) ./ (MaxValue - MinValue);
    end
    % predicts
    test_data = double(test_data);
    [predicted_labels(i), ~, ~] = svmpredict(test_label, test_data, model(i));

end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_labels(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);

Accuracy = (length(find(Category_group1 == 1)) + length(find(Category_group0 == -1))) / Subjects_Quantity;

w_Brain = W_Calculate_SVM(Subjects_Data, Subjects_Label, Pre_Method);


