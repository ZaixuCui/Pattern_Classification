function Accuracy = Ttest_Logistic_QuasiNewton_2group(Subjects_Data, Subjects_Label, P_Value, Pre_Method, ResultantFolder, RetainQuantity)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Label:
%           array of 1 or -1
%
% P_Value:
%           threshold to delete non-important features
%
% Pre_Method:
%           'Scale' or 'Normalzie'
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if ~exist(ResultantFolder, 'dir')
    mkdir(ResultantFolder);
end

[Subjects_Quantity Features_Quantity] = size(Subjects_Data);

t=weka.classifiers.functions.Logistic();
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
    
    % feature selection for training data
    All_Training = [Training_group0_data; Training_group1_data];
    Label = [Training_group0_Label Training_group1_Label];
    
    % T test
    Multiple_Correction.Flag = 'No';
    Multiple_Correction.Method = 'FDR';
    Multiple_Correction.q = 0.05;
    [PValue RetainID{i}] = Ranking_Ttest(All_Training, Label, P_Value, Multiple_Correction);
    if nargin >= 6
        % retain 'RetainQuantity' features with biggest p value
        [~, OriginPos] = sort(PValue, 2, 'ascend');
        RetainID{i} = OriginPos(1:RetainQuantity); 
    end
    All_Training_New = All_Training(:, RetainID{i});
    
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        MeanValue = mean(All_Training_New);
        StandardDeviation = std(All_Training_New);
        [~, columns_quantity] = size(All_Training_New);
        for j = 1:columns_quantity
            if StandardDeviation(j)
                All_Training_New(:, j) = (All_Training_New(:, j) - MeanValue(j)) / StandardDeviation(j);
            end
        end
    elseif strcmp(Pre_Method, 'Scale')
        % Scaling to [0 1]
        MinValue = min(All_Training_New);
        MaxValue = max(All_Training_New);
        [~, columns_quantity] = size(All_Training_New);
        for j = 1:columns_quantity
            All_Training_New(:, j) = (All_Training_New(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
        end
    end

    % SVM classification
    Label = reshape(Label, length(Label), 1);
    Label(find(Label == -1)) = 0;
    All_Training_New = double(All_Training_New);
    X_Y = data(All_Training_New, Label);
    cat = wekaCategoricalData(X_Y);
    t.buildClassifier(cat);
                                                                                                                                                              
    % Ttest
    test_data_New = test_data(RetainID{i});
    % Normalizing
    if strcmp(Pre_Method, 'Normalize')
        % Normalizing
        test_data_New = (test_data_New - MeanValue) ./ StandardDeviation;
    elseif strcmp(Pre_Method, 'Scale')
        % Scale
        test_data_New = (test_data_New - MinValue) ./ (MaxValue - MinValue);
    end
    
    % predicts
    test_data_New = double(test_data_New);
    X_Y = data(test_data_New, test_label);
    dw = wekaCategoricalData(X_Y);
    predicted_labels(i) = t.classifyInstance(dw.instance(0));
    
    % Calculate decision value
    decision_values(i) = [1 test_data_New] * t.coefficients;
    
    % For averaging w
    w{i} = zeros(1, Features_Quantity);
    Logistic_Coefficients = t.coefficients;
    w{i}(RetainID{i}) = Logistic_Coefficients(2:end)';

end

predicted_labels(find(predicted_labels == 0)) = -1;

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == -1);
Category_group1 = predicted_labels(Group1_Index);
Y_group1 = decision_values(Group1_Index);
Category_group0 = predicted_labels(Group0_Index);
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
    
% % Most discriminitive regions
% RetainID_All = RetainID{1};
% for i = 2:Subjects_Quantity
%     RetainID_All = intersect(RetainID_All, RetainID{i});
% end
% % Averaging w
% w_Brain = zeros(1, Features_Quantity);
% for i = 1:Subjects_Quantity
%     w_Brain(RetainID_All) = w_Brain(RetainID_All) + w2{i}(RetainID_All);
% end
% w_Brain = w_Brain / Subjects_Quantity;
% w_Brain = w_Brain / norm(w_Brain);
% save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');

% Calculating weight
RetainID_all = [];
w_all = [];
for i = 1:length(RetainID)
    RetainID_all = [RetainID_all RetainID{i}];
    w_all = [w_all w{i}];
end
Feature_selected_unique = unique(RetainID_all);
for i = 1:length(Feature_selected_unique)
    index = find(RetainID_all == Feature_selected_unique(i));
    Feature_selected(i).ID = Feature_selected_unique(i);
    Feature_selected(i).frequency = length(index);
    Feature_selected(i).averageW = mean(abs(w_all(index)));
end
save([ResultantFolder filesep 'Feature_selected.mat'], 'Feature_selected');

ID_All = [Feature_selected.ID];
Frequency_All = [Feature_selected.frequency];
Weight_All = [Feature_selected.averageW];

% Select features which with frequency bigger than 55 (SubjectQuantity*9/10)
Index = find(Frequency_All >= round(Subjects_Quantity*9/10));
ID_All_2 = ID_All(Index);
Frequency_All_2 = Frequency_All(Index);
Weight_All_2 = Weight_All(Index);

[sort_weight, sort_ind] = sort(abs(Weight_All_2), 2);
ID_All_3 = ID_All_2(sort_ind);
Weight_All_3 = sort_weight;

% Select first 20 features with biggest weight
for i = 1:20
    ID_Final(i) = ID_All_3(end - i + 1);
    Weight_Final(i) = Weight_All_3(end - i + 1);
end

save([ResultantFolder filesep 'ID_Final.mat'], 'ID_Final');
save([ResultantFolder filesep 'Weight_Final.mat'], 'Weight_Final');