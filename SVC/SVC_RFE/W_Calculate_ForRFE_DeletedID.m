function DeletedFeatureID = W_Calculate_ForRFE_DeletedID(Subjects_Data, Subjects_Label, EliminationQuantity, ith, Pre_Method)
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

[Subjects_Quantity Features_Quantity] = size(Subjects_Data);

if strcmp(Pre_Method, 'Normalize')
    %Normalizing
    MeanValue = mean(Subjects_Data);
    StandardDeviation = std(Subjects_Data);
    [rows, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data_New(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
elseif strcmp(Pre_Method, 'Scale')
    % Scaling to [0 1]
    MinValue = min(Subjects_Data);
    MaxValue = max(Subjects_Data);
    [rows, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
    end
end
    
% SVM classification
Subjects_Label = reshape(Subjects_Label, length(Subjects_Label), 1);
Subjects_Data = double(Subjects_Data);
model_All = svmtrain(Subjects_Label, Subjects_Data,'-t 0');
w_Brain = zeros(1, Features_Quantity);
for j = 1 : model_All.totalSV
    w_Brain = w_Brain + model_All.sv_coef(j) * model_All.SVs(j, :);
end
w_Brain = w_Brain;

w_Brain = w_Brain .^ 2;
[RankingResults OriginPos] = sort(w_Brain); % Default is 'ascend'

if length(OriginPos) >= ith * EliminationQuantity
    DeletedFeatureID = OriginPos([ith * EliminationQuantity : -1 : (ith - 1) * EliminationQuantity + 1]);
else
    DeletedFeatureID = OriginPos([end : -1 : (ith - 1) * EliminationQuantity + 1]);
end

