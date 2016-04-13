function [Accuracy Sensitivity Specificity Category] = SVM_2group_Cov(Subjects_Data, Subjects_Label, Pre_Method, Covariates, ResultantFolder)
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
% Covariates:
%           m * n matrix
%           m is the number of subjects
%           n is the number of covariates
%           if no covariates, please set it []
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if nargin >= 4
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity Feature_Quantity] = size(Subjects_Data);

for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    Training_data = Subjects_Data;
    Training_Label = Subjects_Label;
    
    % Select training data and testing data
    test_data = Training_data(i, :);
    test_label = Training_Label(i);
    Training_data(i, :) = [];
    Training_Label(i) = [];
    
    if ~isempty(Covariates)
        Covariates_test = Covariates(i, :);
        Covariates_training = Covariates;
        Covariates_training(i, :) = [];
        [Training_quantity, Covariates_quantity] = size(Covariates_training);
        M = 1;
        for j = 1:Covariates_quantity
            M = M + term(Covariates_training(:, j));
        end
        slm = SurfStatLinMod(Training_data, M);
        
        Training_data_residual = Training_data - repmat(slm.coef(1, :), Training_quantity, 1);
        for j = 1:Covariates_quantity
            Training_data_residual = Training_data_residual - ...
                repmat(Covariates_training(:, j), 1, Feature_Quantity) .* repmat(slm.coef(j + 1, :), Training_quantity, 1);
        end
        Training_data_final = Training_data_residual;
    else
        Training_data_final = Training_data;
    end

    if strcmp(Pre_Method, 'Normalize')
        %Normalizing
        MeanValue = mean(Training_data_final);
        StandardDeviation = std(Training_data_final);
        [rows, columns_quantity] = size(Training_data_final);
        for j = 1:columns_quantity
            Training_data_final(:, j) = (Training_data_final(:, j) - MeanValue(j)) / StandardDeviation(j);
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(Training_data_final);
        MaxValue = max(Training_data_final);
        [rows, columns_quantity] = size(Training_data_final);
        for j = 1:columns_quantity
            Training_data_final(:, j) = (Training_data_final(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end

    % SVM classification
    Training_Label = reshape(Training_Label, length(Training_Label), 1);
    Training_data_final = double(Training_data_final);
    model(i) = svmtrain(Training_Label, Training_data_final,'-t 0');
    
    if isempty(Covariates)
        test_data_final = test_data;
    else
        test_data_residual = test_data - slm.coef(1, :);
        for j = 1:Covariates_quantity
            test_data_residual = test_data_residual - repmat(Covariates_test(j), 1, Feature_Quantity) .* slm.coef(j + 1, :);
        end
        test_data_final = test_data_residual;
    end

    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data_final = (test_data_final - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data_final = (test_data_final - MinValue) ./ (MaxValue - MinValue);
    end

    % predicts
    test_data_final = double(test_data_final);
    [predicted_labels(i), accuracy, tmp] = svmpredict(test_label, test_data_final, model(i));
    
    w{i} = zeros(1, Feature_Quantity);
    for j = 1 : model(i).totalSV
        w{i} = w{i} + model(i).sv_coef(j) * model(i).SVs(j, :);
    end
    decision_values(i) = w{i} * test_data_final' - model(i).rho;
    Distance_svm(i) = decision_values(i) / norm(w{i});
    
end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_labels(Group1_Index);
Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);
Y_group0 = decision_values(Group0_Index);
% Calculating distance for svm
Dis_group1_svm = Distance_svm(Group1_Index);
Dis_group0_svm = Distance_svm(Group0_Index);
save([ResultantFolder filesep 'Dis_svm.mat'], 'Dis_group1_svm', 'Dis_group0_svm');

if nargin >= 5
    save([ResultantFolder filesep 'Y.mat'], 'Y_group1', 'Y_group0');
    save([ResultantFolder filesep 'Category.mat'], 'Category_group1', 'Category_group0');
end

Category.Category_group0 = Category_group0;
Category.Category_group1 = Category_group1;

group0_Wrong_ID = find(Category_group0 == 1);
group0_Wrong_Quantity = length(group0_Wrong_ID);
group1_Wrong_ID = find(Category_group1 == -1);
group1_Wrong_Quantity = length(group1_Wrong_ID);

Accuracy = (Subjects_Quantity - group0_Wrong_Quantity - group1_Wrong_Quantity) / Subjects_Quantity;
Sensitivity = (length(Group0_Index) - group0_Wrong_Quantity) / length(Group0_Index);
Specificity = (length(Group1_Index) - group1_Wrong_Quantity) / length(Group1_Index);

if nargin >= 5
    disp(['group0: ' num2str(group0_Wrong_Quantity) ' subjects are wrong ' mat2str(group0_Wrong_ID) ]);
    disp(['group1: ' num2str(group1_Wrong_Quantity) ' subjects are wrong ' mat2str(group1_Wrong_ID) ]);
    save([ResultantFolder filesep 'WrongInfo.mat'], 'group0_Wrong_Quantity', 'group0_Wrong_ID', 'group1_Wrong_Quantity', 'group1_Wrong_ID');
    disp(['Accuracy is ' num2str(Accuracy) ' !']);
    save([ResultantFolder filesep 'Accuracy.mat'], 'Accuracy');
    disp(['Sensitivity is ' num2str(Sensitivity) ' !']);
    save([ResultantFolder filesep 'Sensitivity.mat'], 'Sensitivity');
    disp(['Specificity is ' num2str(Specificity) ' !']);
    save([ResultantFolder filesep 'Specificity.mat'], 'Specificity');
    PPV = length(find(Category_group0 == -1)) / length(find([Category_group0 Category_group1] == -1));
    save([ResultantFolder filesep 'PPV.mat'], 'PPV');
    NPV = length(find(Category_group1 == 1)) / length(find([Category_group0 Category_group1] == 1));
    save([ResultantFolder filesep 'NPV.mat'], 'NPV');

    % Calculating w
    W_Calculate_SVM_Cov(Subjects_Data, Subjects_Label, Pre_Method, Covariates, ResultantFolder);
end