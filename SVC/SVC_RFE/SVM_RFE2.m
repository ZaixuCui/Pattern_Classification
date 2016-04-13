function RetainID = SVM_RFE2(Subjects_Data, Subjects_Label, EliminationQuantity)

[~, FeatureQuantity] = size(Subjects_Data);
Feature_Index = [1:FeatureQuantity];

AccuracyBefore = 0;
Accuracy = 0.001;
RFE_Quantity = 0;

DeletedSetID = [];
while AccuracyBefore <= Accuracy
    
    RFE_Quantity = RFE_Quantity + 1;
    disp(['The ' num2str(RFE_Quantity) 'th SVM!']);
    
    Subjects_Data_New = Subjects_Data(:, Feature_Index);
    AccuracyBefore = Accuracy;
    
    [Accuracy, w_Brain] = SVM_2group_ForRFE_Accuracy(Subjects_Data_New, Subjects_Label, 'Normalize');
    [~, Origin_Index] = sort(abs(w_Brain), 2);
    
    if Accuracy == 1 | length(Origin_Index) < EliminationQuantity
        break;
    end
    
    DeletedIndex = Origin_Index(1:EliminationQuantity);
    
    DeletedSetID = [DeletedSetID Feature_Index(DeletedIndex)];
    
    Subjects_Data_New(:, DeletedIndex) = [];
    Feature_Index(:, DeletedIndex) = [];
    
end

RetainID = Feature_Index;





