function Ttest_MLDA_2group(Subjects_Data, Subjects_Label, P_Value, Pre_Method, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Training_Label:
%           array of 1 or -1
%
% Component_Quantity:
%           single
%           '-1' means using all components required from PCA
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

Subjects_Quantity = length(Subjects_Label);

for i = 1:Subjects_Quantity
    
    disp(['The ' num2str(i) ' iteration!']);
    
    All_Training = Subjects_Data;
    Training_Label = Subjects_Label;
    
    % Select training data and testing data
    test_data = All_Training(i, :);
    test_Training_Label = Training_Label(i);
    All_Training(i, :) = [];
    Training_Label(i) = [];
    
    % T test
    [PValues RetainID] = Ranking_Ttest(All_Training, Training_Label, P_Value);
    All_Training_New = All_Training(:, RetainID);
    
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        MeanValue = mean(All_Training_New);
        StandardDeviation = sqrt(var(All_Training_New));
        [rows, columns_quantity] = size(All_Training_New);
        for j = 1:columns_quantity
            if StandardDeviation(j)
                All_Training_New(:, j) = (All_Training_New(:, j) - MeanValue(j)) / StandardDeviation(j);
            end
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(All_Training_New);
        MaxValue = max(All_Training_New);
        [rows, columns_quantity] = size(All_Training_New);
        for j = 1:columns_quantity
            All_Training_New(:, j) = (All_Training_New(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end
    
    % SVM classification
    Training_Label = reshape(Training_Label, length(Training_Label), 1);
    All_Training_New = double(All_Training_New);
    model(i) = MLDA_train(All_Training_New, Training_Label);
    
    % group1 is 1, group0 is -1
    % Ttest
    test_data_New = test_data(RetainID);
    % Normalizing
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data_New = (test_data_New - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data_New = (test_data_New - MinValue) ./ (MaxValue - MinValue);
    end

    % predicts
    [predicted_Labels(i), decision_values(i)] = MLDA_predict(model(i), test_data_New, test_Training_Label);

end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_Labels(Group1_Index);
Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_Labels(Group0_Index);
Y_group0 = decision_values(Group0_Index);

save([ResultantFolder filesep 'Y.mat'], 'Y_group1', 'Y_group0');
save([ResultantFolder filesep 'Category.mat'], 'Category_group1', 'Category_group0');

group0_Wrong_ID = find(Category_group0 == 1);
group0_Wrong_Quantity = length(group0_Wrong_ID);
group1_Wrong_ID = find(Category_group1 == -1);
group1_Wrong_Quantity = length(group1_Wrong_ID);
disp(['group0: ' num2str(group0_Wrong_Quantity) ' subjects are wrong ' mat2str(group0_Wrong_ID) ]);
disp(['group1: ' num2str(group1_Wrong_Quantity) ' subjects are wrong ' mat2str(group1_Wrong_ID) ]);
save([ResultantFolder filesep 'WrongInfo.mat'], 'group0_Wrong_Quantity', 'group0_Wrong_ID', 'group1_Wrong_Quantity', 'group1_Wrong_ID');
Accuracy = (Subjects_Quantity - group0_Wrong_Quantity - group1_Wrong_Quantity) / Subjects_Quantity;
disp(['Accuracy is ' num2str(Accuracy) ' !']);
save([ResultantFolder filesep 'Accuracy.mat'], 'Accuracy');
group0_Quantity = length(find(Subjects_Label == -1));
group1_Quantity = length(find(Subjects_Label == 1));
Sensitivity = (group0_Quantity - group0_Wrong_Quantity) / group0_Quantity;
Specificity = (group1_Quantity - group1_Wrong_Quantity) / group1_Quantity;
disp(['Sensitivity is ' num2str(Sensitivity) ' !']);
save([ResultantFolder filesep 'Sensitivity.mat'], 'Sensitivity');
disp(['Specificity is ' num2str(Specificity) ' !']);
save([ResultantFolder filesep 'Specificity.mat'], 'Specificity');
PPV = length(find(Category_group0 == -1)) / length(find([Category_group0 Category_group1] == -1));
save([ResultantFolder filesep 'PPV.mat'], 'PPV');
NPV = length(find(Category_group1 == 1)) / length(find([Category_group0 Category_group1] == 1));
save([ResultantFolder filesep 'NPV.mat'], 'NPV');

% % Averaging w
% w_average = 0;
% for i = 1:Subjects_Quantity
%     w_average = w_average + model(i).w;
% end
% w_average = w_average / Subjects_Quantity;
% save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');