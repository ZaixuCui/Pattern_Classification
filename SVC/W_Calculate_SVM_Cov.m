
function [w_Brain, model_All] = W_Calculate_SVM_Cov(Subjects_Data, Subjects_Label, Pre_Method, Covariates, ResultantFolder)

if nargin >= 5
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_quantity, Features_Quantity] = size(Subjects_Data);

if ~isempty(Covariates)
    [~, Covariates_quantity] = size(Covariates);
    M = 1;
    for j = 1:Covariates_quantity
        M = M + term(Covariates(:, j));
    end
    slm = SurfStatLinMod(Subjects_Data, M);
    
    Subjects_data_residual = Subjects_Data - repmat(slm.coef(1, :), Subjects_quantity, 1);
    for j = 1:Covariates_quantity
        Subjects_data_residual = Subjects_data_residual - ...
            repmat(Covariates(:, j), 1, Features_Quantity) .* repmat(slm.coef(j + 1, :), Subjects_quantity, 1);
    end
    Subjects_data_final = Subjects_data_residual;
else
    Subjects_data_final = Subjects_Data;
end

if strcmp(Pre_Method, 'Normalize')
    %Normalizing
    MeanValue = mean(Subjects_data_final);
    StandardDeviation = std(Subjects_data_final);
    [~, columns_quantity] = size(Subjects_data_final);
    for j = 1:columns_quantity
        Subjects_data_final(:, j) = (Subjects_data_final(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
elseif strcmp(Pre_Method, 'Scale')
    % Scaling to [0 1]
    MinValue = min(Subjects_data_final);
    MaxValue = max(Subjects_data_final);
    [~, columns_quantity] = size(Subjects_data_final);
    for j = 1:columns_quantity
        Subjects_data_final(:, j) = (Subjects_data_final(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
    end
end
    
% SVM classification
Subjects_Label = reshape(Subjects_Label, length(Subjects_Label), 1);
Subjects_data_final = double(Subjects_data_final);
model_All = svmtrain(Subjects_Label, Subjects_data_final,'-t 0');
w_Brain = zeros(1, Features_Quantity);
for j = 1 : model_All.totalSV
    w_Brain = w_Brain + model_All.sv_coef(j) * model_All.SVs(j, :);
end
save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');
% decision_values = w_Brain * Subjects_Data' - model_All.rho;
% for i = 1:Subjects_Quantity
%     tmp = Subjects_Data;
%     tmp(i, :) = [];
%     Mean_Data_Leaveoneout(i, :) = mean(tmp);
% end
% Mean_Data_Leaveoneout_Sort = sort(Mean_Data_Leaveoneout);
% Mean_Data_Median = Mean_Data_Leaveoneout_Sort(48, :);
% w_Brain = w_Brain .* Mean_Data_Median;

% w_Brain = w_Brain / norm(w_Brain);

% % if nargin >= 4
% %     save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');
% % end
% % 
% % [orig_W ind] = sort(abs(w_Brain), 2, 'descend');
% % for i = 1:50
% %     Decision_values = Subjects_Data(:, ind([1:i])) * w_Brain(ind([1:i]))' - model_All.rho;
% %     Decision_Label(find(Decision_values > 0)) = -1;
% %     Decision_Label(find(Decision_values <= 0)) = 1;
% %     Accuracy(i) = (length(find(Decision_Label(1:28) == -1)) + length(find(Decision_Label(29:61) == 1))) / 61;
% % end
% % 
% % plot(Accuracy)