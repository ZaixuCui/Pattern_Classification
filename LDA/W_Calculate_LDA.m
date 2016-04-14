
function w_Brain = W_Calculate_LDA(Subjects_Data, Subjects_Label, Pre_Method, ResultantFolder)

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

% Normalizing
if strcmp(Pre_Method, 'Normalize')
    %Normalizing
    MeanValue = mean(Subjects_Data);
    StandardDeviation = std(Subjects_Data);
    [rows, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
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


% LDA classification
Subjects_Label = reshape(Subjects_Label, length(Subjects_Label), 1);
Subjects_Data = double(Subjects_Data);
model_All = LDA_train(Subjects_Data, Subjects_Label);
w_Brain = model_All.w / norm(model_All.w);
save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');