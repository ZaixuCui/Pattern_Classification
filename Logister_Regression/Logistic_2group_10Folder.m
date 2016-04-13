function Logistic_2group_10Folder(Subjects_Data, Subjects_Label, alpha, Pre_Method, ResultantFolder)
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

[Subjects_Quantity, Feature_Quantity] = size(Subjects_Data);
[Splited_Data, Splited_Data_Label, Origin_ID_Cell] = Split_NFolders(Subjects_Data, Subjects_Label, 10);

predicted_labels = [];
decision_values = [];
for i = 1:10
    
    disp(['The ' num2str(i) ' iteration!']);
    
    % Select training data and testing data
    test_label = Splited_Data_Label{i};
    test_data = Splited_Data{i};
    
    Training_all_data = [];
    Label = [];
    for j = 1:10
        if j == i
            continue;
        end
        Training_all_data = [Training_all_data; Splited_Data{j}];
        Label = [Label; Splited_Data_Label{j}];
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

    % Logistic classification
    Label = reshape(Label, length(Label), 1);
    Training_all_data = double(Training_all_data);
    theta = Logistic_train(Training_all_data, Label, alpha);

    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        MeanValue_New = repmat(MeanValue, length(test_label), 1);
        StandardDeviation_New = repmat(StandardDeviation, length(test_label), 1);
        test_data = (test_data - MeanValue_New) ./ StandardDeviation_New;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        MaxValue_New = repmat(MaxValue, length(test_label), 1);
        MinValue_New = repmat(MinValue, length(test_label), 1);
        test_data = (test_data - MinValue_New) ./ (MaxValue_New - MinValue_New);
    end

    % predicts
    test_data = double(test_data);
    [predicted_labels_tmp, decision_values_tmp] = Logistic_predict(theta, test_data, test_label);
    predicted_labels = [predicted_labels predicted_labels_tmp];
    decision_values = [decision_values decision_values_tmp];
    
end

Origin_ID = [];
for i = 1:length(Origin_ID_Cell)
    Origin_ID = [Origin_ID; Origin_ID_Cell{i}];
end

Group1_Index = find(Subjects_Label(Origin_ID)==1);
Group0_Index = find(Subjects_Label(Origin_ID)==0);
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
