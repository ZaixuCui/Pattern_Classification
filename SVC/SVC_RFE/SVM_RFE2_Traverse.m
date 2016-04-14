function RetainID = SVM_RFE2_Traverse(Subjects_Data, Subjects_Label, EliminationQuantity)

[~, FeatureQuantity] = size(Subjects_Data);
Feature_Index = [1:FeatureQuantity];

AccuracyBefore = 0;
Accuracy = 0.001;
RFE_Quantity = 0;

DeletedSetID = [];
while ~isempty(Feature_Index)
    
    RFE_Quantity = RFE_Quantity + 1;
    disp(['The ' num2str(RFE_Quantity) 'th SVM!']);
    
    Subjects_Data_New = Subjects_Data(:, Feature_Index);
    
    [Accuracy, w_Brain] = SVM_2group_ForRFE_Accuracy(Subjects_Data_New, Subjects_Label, 'Normalize');
    [~, Origin_Index] = sort(abs(w_Brain), 2);
    if length(Origin_Index) < EliminationQuantity
        DeletedIndex = Origin_Index;
    else
        DeletedIndex = Origin_Index(1:EliminationQuantity);
    end
    
    DeletedSetID = [DeletedSetID Feature_Index(DeletedIndex)];
    
    Subjects_Data_New(:, DeletedIndex) = [];
    Feature_Index(:, DeletedIndex) = [];
    
    if AccuracyBefore <= Accuracy
        RetainID = Feature_Index;
        AccuracyBefore = Accuracy;
    end
    
end







