function Logistic_2group(Subjects_Data, Subjects_Label, alpha, Pre_Method, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 0 or 1
%
% Pre_Method:
%           'Normalize' or 'Scale'
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[Subjects_Quantity Feature_Quantity] = size(Subjects_Data);

theta_init = zeros(Feature_Quantity, 1);
funObj = @(w)LogisticLoss(w,X,y);
options.Method = 'pnewton0';
options.precFunc = [];
for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    Training_all_data = Subjects_Data;
    Training_all_Label = Subjects_Label;
    
    % Select training data and testing data
    test_data = Training_all_data(i, :);
    test_label = Training_all_Label(i);
    Training_all_data(i, :) = [];
    Training_all_Label(i) = [];

    if strcmp(Pre_Method, 'Normalize')
        %Normalizing
        MeanValue = mean(Training_all_data);
        StandardDeviation = std(Training_all_data);
        [rows, columns_quantity] = size(Training_all_data);
        for j = 1:columns_quantity
            Training_all_data(:, j) = (Training_all_data(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(Training_all_data);
        MaxValue = max(Training_all_data);
        [rows, columns_quantity] = size(Training_all_data);
        for j = 1:columns_quantity
            Training_all_data(:, j) = (Training_all_data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end

    % Logistic classification
    Training_all_Label = reshape(Training_all_Label, length(Training_all_Label), 1);
    Training_all_data = double(Training_all_data);
%     theta = Logistic_train(Training_all_data, Training_all_Label, alpha);
    theta = minFunc(@LogisticLoss, theta_init, options, Training_all_data, Training_all_Label);

    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data = (test_data - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data = (test_data - MinValue) ./ (MaxValue - MinValue);
    end

    % predicts
    test_data = double(test_data);
    [predicted_labels(i), decision_values(i)] = Logistic_predict(theta, test_data, test_label);
    
end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == 0);
Category_group1 = predicted_labels(Group1_Index);
Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);
Y_group0 = decision_values(Group0_Index);

save([ResultantFolder filesep 'Y.mat'], 'Y_group1', 'Y_group0');
save([ResultantFolder filesep 'Category.mat'], 'Category_group1', 'Category_group0');

group0_Wrong_ID = find(Category_group0 == 1);
group0_Wrong_Quantity = length(group0_Wrong_ID);
group1_Wrong_ID = find(Category_group1 == 0);
group1_Wrong_Quantity = length(group1_Wrong_ID);
disp(['group0: ' num2str(group0_Wrong_Quantity) ' subjects are wrong ' mat2str(group0_Wrong_ID) ]);
disp(['group1: ' num2str(group1_Wrong_Quantity) ' subjects are wrong ' mat2str(group1_Wrong_ID) ]);
save([ResultantFolder filesep 'WrongInfo.mat'], 'group0_Wrong_Quantity', 'group0_Wrong_ID', 'group1_Wrong_Quantity', 'group1_Wrong_ID');
Accuracy = (Subjects_Quantity - group0_Wrong_Quantity - group1_Wrong_Quantity) / Subjects_Quantity;
disp(['Accuracy is ' num2str(Accuracy) ' !']);
save([ResultantFolder filesep 'Accuracy.mat'], 'Accuracy');

% Calculating w
W_Calculate_Logistic(Subjects_Data, Subjects_Label, alpha, Pre_Method, ResultantFolder);
