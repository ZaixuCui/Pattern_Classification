function Ttest_MLDA_2group_PSelection_SGE_child(SubjectsData_Path, Subjects_Label, i, P_Value_Range, Pre_Method, ResultantFolder)
%
% SubjectData_Path:
%           path of .mat file containing a m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 1 or -1
%
% i:
%           the ith folder
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

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end
load(SubjectsData_Path);
Subjects_Quantity = length(Subjects_Label);
    
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

for j = 1:length(P_Value_Range)
    Accuracy(j) = Ttest_MLDA_2group_ACC(All_Training, Label, P_Value_Range(j), Pre_Method);
end
P_Value_BestSet = P_Value_Range(find(Accuracy == max(Accuracy)));
P_Value_Final = P_Value_BestSet(1);
save([ResultantFolder filesep 'P_Final_' num2str(i) '.mat'], 'P_Value_Final');

% T test
[PValue RetainID] = Ranking_Ttest(All_Training, Label, P_Value_Final);
All_Training_New = All_Training(:, RetainID);
save([ResultantFolder filesep 'RetainID_' num2str(i) '.mat'], 'RetainID');

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
model = MLDA_train(All_Training_New, Label);

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
[predicted_label, decision_value] = MLDA_predict(model, test_data_New, test_label);
save([ResultantFolder filesep 'predicted_labels_' num2str(i) '.mat'], 'predicted_label');
save([ResultantFolder filesep 'decision_values_' num2str(i) '.mat'], 'decision_value');

% For averaging w
w = model.w / norm(model.w);
save([ResultantFolder filesep 'w_' num2str(i) '.mat'], 'w');


