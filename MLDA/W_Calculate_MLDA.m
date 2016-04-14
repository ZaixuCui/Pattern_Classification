
function w_Brain = W_Calculate_MLDA(Subjects_Data, Subjects_Label, ResultantFolder)

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[Subjects_Quantity Features_Quantity] = size(Subjects_Data);

% Normalizing
MeanValue = mean(Subjects_Data);
StandardDeviation = sqrt(var(Subjects_Data));
[rows, columns_quantity] = size(Subjects_Data);
for j = 1:columns_quantity
    if StandardDeviation(j)
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
end

% SVM classification
Subjects_Label = reshape(Subjects_Label, length(Subjects_Label), 1);
Subjects_Data = double(Subjects_Data);
model_All = MLDA_train(Subjects_Data, Subjects_Label);
w_Brain = model_All.w';
w_Brain = w_Brain / norm(w_Brain);
save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');