
function RetainID = SVM_RFE_Main_Accuracy(SubjectsData, SubjectsLabel, EliminationQuantity)

%
% FeaturesListOrder is from important to non-important
%
% After acquiring features order, using accuracy to get best set of
% features
%

if isempty(EliminationQuantity)
    [FeaturesListOrder RFE_Quantity] = SVM_RFE(SubjectsData, SubjectsLabel);
else
    [FeaturesListOrder RFE_Quantity] = SVM_RFE(SubjectsData, SubjectsLabel, EliminationQuantity);
end
   
Subjects_Quantity = length(SubjectsLabel);
Max_id = 1;
Accuracy_before = 0;
for j = 1:RFE_Quantity
    
    if isempty(EliminationQuantity)
        FeatureID = FeaturesListOrder([1 : 2^(j - 1)]);
    else
        if j * EliminationQuantity < length(FeaturesListOrder)
            FeatureID = FeaturesListOrder([1 : j * EliminationQuantity]);
        else
            FeatureID = FeaturesListOrder;
        end
    end
    Data_tmp = SubjectsData(:, FeatureID);
    Accuracy = SVM_2group_ForRFE_Accuracy(Data_tmp, SubjectsLabel, 'Normalize');
    clear Data_tmp;
    if Accuracy > Accuracy_before
        Max_id = j;
        Accuracy_before = Accuracy;
    end
    
end
RetainID = FeaturesListOrder(1:Max_id);