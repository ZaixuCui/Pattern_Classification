
function w_Brain = W_Calculate_PCA_MLDA(Subjects_Data, Subjects_Label, type, ResultantFolder, Parameters)

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
[COEFF, SCORE, latent, tsquare] = princomp(Subjects_Data, 'econ');

if strcmp(type, 'ComponentQuantity')
    ComponentQuantity = Parameters;
elseif strcmp(type, 'CumulativeVariation')
    ComponentQuantity = length(latent);
    while 1
        if sum(latent(1:ComponentQuantity)) / sum(latent) <= Parameters
            break;
        else
            ComponentQuantity = ComponentQuantity - 1;
        end
    end
elseif strcmp(type, 'BiggerThanMean')
    tmp = find(latent >= mean(latent));
    ComponentQuantity = tmp(end);
end

clear PCA_All;
for j = 1:Subjects_Quantity
    PCA_All(j, :) = SCORE(j, 1:ComponentQuantity);
end
% SVM classification
Subjects_Label = reshape(Subjects_Label, length(Subjects_Label), 1);
PCA_All = double(PCA_All);
model_All = MLDA_train(PCA_All, Subjects_Label);
w_Brain = model_All.w' * COEFF';
w_Brain = w_Brain / norm(w_Brain);
save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');