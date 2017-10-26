function [Accuracy Sensitivity Specificity Category] = SVM_2group(Subjects_Data, Subjects_Label, Pre_Method, ResultantFolder, Permutation_Flag)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of -1 or 1
%
% Pre_Method:
%           'Normalize' or 'Scale'
%
% ResultantFolder:
%           the path of folder storing resultant files
%
% Permutation_Flag:
%           0 or 1; if 1, doing permutation, that is randomizing the subjects' label and then classifying the two groups
%

if nargin >= 4
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity Feature_Quantity] = size(Subjects_Data);

for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    Training_all_data = Subjects_Data;
    Training_all_Label = Subjects_Label;
    
    % Select training data and testing data
    test_data = Training_all_data(i, :);
    test_label = Training_all_Label(i);
    Training_all_data(i, :) = [];
    Training_all_Label(i) = [];
    
    if Permutation_Flag
        Rand_ID = randperm(length(Training_all_Label));
        Training_all_Label = Training_all_Label(Rand_ID);
    end

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

    % SVM classification
    Training_all_Label = reshape(Training_all_Label, length(Training_all_Label), 1);
    Training_all_data = double(Training_all_data);
    model(i) = svmtrain(Training_all_Label, Training_all_data,'-t 0');

    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data = (test_data - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data = (test_data - MinValue) ./ (MaxValue - MinValue);
    end

    % predicts
    test_data = double(test_data);
    [predicted_labels(i), accuracy, tmp] = svmpredict(test_label, test_data, model(i));
    
    w{i} = zeros(1, Feature_Quantity);
    for j = 1 : model(i).totalSV
        w{i} = w{i} + model(i).sv_coef(j) * model(i).SVs(j, :);
    end
    decision_values(i) = w{i} * test_data' - model(i).rho;
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

if nargin >= 4
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

if nargin >= 4
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
    W_Calculate_SVM(Subjects_Data, Subjects_Label, Pre_Method, ResultantFolder);
end
