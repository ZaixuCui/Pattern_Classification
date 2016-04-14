
function w_Brain = W_Calculate_Pearson_SVM(Subjects_Data, Subjects_Label, Threshold, Pre_Method, ResultantFolder, RetainQuantity)

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[~, Features_Quantity] = size(Subjects_Data);

if nargin >= 6
    [~, RetainID, ~] = Ranking_Pearson(Subjects_Data, Subjects_Label, 'normal', RetainQuantity);
else
    [~, RetainID, ~] = Ranking_Pearson(Subjects_Data, Subjects_Label, 'normal', '', Threshold);
end
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
w_Brain = zeros(1, Features_Quantity);
for j = 1 : model_All.totalSV
    w_Brain(RetainID) = w_Brain(RetainID) + model_All.sv_coef(j) * model_All.SVs(j, :);
end
w_Brain = w_Brain / norm(w_Brain);
save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');