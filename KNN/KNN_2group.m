function [Accuracy Sensitivity Specificity Category] = KNN_2group(Subjects_Data, Subjects_Label, Pre_Method, ResultantFolder)
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

if nargin >= 4
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity Feature_Quantity] = size(Subjects_Data);

t = weka.classifiers.lazy.IBk();
tmp=wekaArgumentString({'-K', 8});
% tmp=wekaArgumentString({'-K', round(sqrt(Subjects_Quantity - 1))});
t.setOptions(tmp);
for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    Subjects_Data_tmp = Subjects_Data;
    Subjects_Label_tmp = Subjects_Label;
    % Select training data and testing data
    test_label = Subjects_Label_tmp(i);
    test_data = Subjects_Data_tmp(i, :);
    
    Subjects_Label_tmp(i) = [];
    Subjects_Data_tmp(i, :) = [];
    Training_group1_Index = find(Subjects_Label_tmp == 1);
    Training_group0_Index = find(Subjects_Label_tmp == -1);
    Training_group1_data = Subjects_Data_tmp(Training_group1_Index, :);
    Training_group0_data = Subjects_Data_tmp(Training_group0_Index, :);
    Training_group1_Label = Subjects_Label_tmp(Training_group1_Index);
    Training_group0_Label = Subjects_Label_tmp(Training_group0_Index);
    
    Training_all_data = [Training_group1_data; Training_group0_data];
    Label = [Training_group1_Label Training_group0_Label];

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

    Label = reshape(Label, length(Label), 1);
    Training_all_data = double(Training_all_data);
    
    X_Y = data(Training_all_data, Label);
    cat = wekaCategoricalData(X_Y);
    t.buildClassifier(cat);

    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data = (test_data - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data = (test_data - MinValue) ./ (MaxValue - MinValue);
    end

    % predicts
    test_data = double(test_data);
    X_Y = data(test_data, test_label);
    dw = wekaCategoricalData(X_Y);
    predicted_labels(i) = t.classifyInstance(dw.instance(0));
    group_Prob(i,:)=t.distributionForInstance(dw.instance(0));
end

predicted_labels(find(~predicted_labels)) = -1;

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_labels(Group1_Index);
% Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);
% Y_group0 = decision_values(Group0_Index);

if nargin >= 4
    % save([ResultantFolder filesep 'Y.mat'], 'Y_group1', 'Y_group0');
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

    % Calculating w
    % W_Calculate_Logistic(Subjects_Data, Subjects_Label, alpha, Pre_Method, ResultantFolder);
end
