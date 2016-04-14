
function w_Brain = W_Calculate_PCA_SVM(Subjects_Data, Subjects_Label, type, ResultantFolder, Parameters)

%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 1 or -1
%
% type:
%           'ComponentQuantity'
%           'CumulativeVariation'
%           'BiggerThanMean'
% 
% Parameters:
%           the quantity of components when type is 'ComponentQuantity';
%           the proportion of cumulative variation when type is
%           'CumulativeVariation'
%           don't need when type is 'BiggerThanMean'
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[Subjects_Quantity Features_Quantity] = size(Subjects_Data);

% Normalizing
MeanValue = mean(Subjects_Data);
StandardDeviation = std(Subjects_Data);
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
model_All = svmtrain(Subjects_Label, PCA_All,'-t 0');
w_Brain_Component = zeros(1, ComponentQuantity);
for j = 1 : model_All.totalSV
    w_Brain_Component = w_Brain_Component + model_All.sv_coef(j) * model_All.SVs(j, :);
end
w_Brain_PCA = zeros(1, length(latent));
w_Brain_PCA(1:ComponentQuantity) = w_Brain_PCA(1:ComponentQuantity) + w_Brain_Component;
w_Brain = w_Brain_PCA * COEFF';
w_Brain = w_Brain / norm(w_Brain);
save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');