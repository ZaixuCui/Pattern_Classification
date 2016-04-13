function Accuracy = Ttest_SVM_2group_RetainQuantity(Subjects_Data, Subjects_Label, RetainQuantity, Pre_Method)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 1 or -1
%
% P_Value:
%           threshold to delete non-important features
%
% Pre_Method:
%           'Scale' or 'Normalzie'
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
    
    % feature selection for training data
    All_Training = [Training_group1_data; Training_group0_data];
    Label = [Training_group1_Label Training_group0_Label];
    
    % T test
    [PValue RetainID] = Ranking_Ttest(All_Training, Label, 0.05);
    % retain 'RetainQuantity' features with biggest p value
    [~, OriginPos] = sort(PValue, 2, 'ascend');
    RetainID = OriginPos(1:RetainQuantity);
    All_Training_New = All_Training(:, RetainID);
    
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

    % SVM classification
    Label = reshape(Label, length(Label), 1);
    All_Training_New = double(All_Training_New);
    model(i) = svmtrain(Label, All_Training_New,'-t 0');
                                                                                                                                                              
    % Ttest
    test_data_New = test_data(RetainID);
    % Normalizing
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data_New = (test_data_New - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data_New = (test_data_New - MinValue) ./ (MaxValue - MinValue);
    end
    
    % predicts
    test_data_New = double(test_data_New);
    [predicted_labels(i), accuracy, tmp] = svmpredict(test_label, test_data_New, model(i));

end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_labels(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);

group0_Wrong_ID = find(Category_group0 == 1);
group0_Wrong_Quantity = length(group0_Wrong_ID);
group1_Wrong_ID = find(Category_group1 == -1);
group1_Wrong_Quantity = length(group1_Wrong_ID);
Accuracy = (Subjects_Quantity - group0_Wrong_Quantity - group1_Wrong_Quantity) / Subjects_Quantity;

