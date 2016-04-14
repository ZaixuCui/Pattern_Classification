
function RetainID = SVM_RFE_Main_Quantity(SubjectsData, SubjectsLabel, EliminationQuantity, RetainQuantity)

%
% FeaturesListOrder is from important to non-important
%
% After acquiring features order, and then reatain features with a fixed
% quantity
%

if isempty(EliminationQuantity)
    FeaturesListOrder = SVM_RFE(SubjectsData, SubjectsLabel);
else
    FeaturesListOrder = SVM_RFE(SubjectsData, SubjectsLabel, EliminationQuantity);
end
   
RetainID = FeaturesListOrder(1:RetainQuantity);