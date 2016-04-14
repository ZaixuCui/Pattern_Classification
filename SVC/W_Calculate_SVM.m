
function [w_Brain, model_All] = W_Calculate_SVM(Subjects_Data, Subjects_Label, Pre_Method, ResultantFolder)

if nargin >= 4
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity, Features_Quantity] = size(Subjects_Data);

if strcmp(Pre_Method, 'Normalize')
    %Normalizing
    MeanValue = mean(Subjects_Data);
    StandardDeviation = std(Subjects_Data);
    [~, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
elseif strcmp(Pre_Method, 'Scale')
    % Scaling to [0 1]
    MinValue = min(Subjects_Data);
    MaxValue = max(Subjects_Data);
    [~, columns_quantity] = size(Subjects_Data);
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
w_Brain = w_Brain / norm(w_Brain);
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