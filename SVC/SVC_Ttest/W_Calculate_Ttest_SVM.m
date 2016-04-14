
function [Accuracy w_Brain] = W_Calculate_Ttest_SVM(Subjects_Data, Subjects_Label, P_Value, Pre_Method, ResultantFolder)

if nargin >= 5
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[~, Features_Quantity] = size(Subjects_Data);

% T test
Multiple_Correction.Flag = 'No';
Multiple_Correction.Method = 'FDR';
Multiple_Correction.q = 0.05;
[~, RetainID] = Ranking_Ttest(Subjects_Data, Subjects_Label, P_Value, Multiple_Correction);
% [PValue RetainID] = Ranking_Ttest(Subjects_Data, Subjects_Label, P_Value);
Subjects_Data_New = Subjects_Data(:, RetainID);

if strcmp(Pre_Method, 'Normalize')
    %Normalizing
    MeanValue = mean(Subjects_Data_New);
    StandardDeviation = std(Subjects_Data_New);
    [~, columns_quantity] = size(Subjects_Data_New);
    for j = 1:columns_quantity
        Subjects_Data_New(:, j) = (Subjects_Data_New(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
elseif strcmp(Pre_Method, 'Scale')
    % Scaling to [0 1]
    MinValue = min(Subjects_Data_New);
    MaxValue = max(Subjects_Data_New);
    [~, columns_quantity] = size(Subjects_Data_New);
    for j = 1:columns_quantity
        Subjects_Data_New(:, j) = (Subjects_Data_New(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
    end
end
    
% SVM classification
Subjects_Label = reshape(Subjects_Label, length(Subjects_Label), 1);
Subjects_Data_New = double(Subjects_Data_New);
model_All = svmtrain(Subjects_Label, Subjects_Data_New,'-t 0');

[~, tmp, ~] = svmpredict(Subjects_Label, Subjects_Data_New, model_All);
Accuracy = tmp(1);

w_Brain = zeros(1, Features_Quantity);
for j = 1 : model_All.totalSV
    w_Brain(RetainID) = w_Brain(RetainID) + model_All.sv_coef(j) * model_All.SVs(j, :);
end
w_Brain = w_Brain / norm(w_Brain);
if nargin >= 5
    save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');
end