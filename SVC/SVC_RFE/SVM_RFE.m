function [FeaturesListOrder RFE_Quantity] = SVM_RFE(Subjects_Data, Subjects_Label, EliminationQuantity)

[~, FeatureQuantity] = size(Subjects_Data);
Feature_Index = [1:FeatureQuantity];

AccuracyBefore = 0;
RFE_Quantity = 0;
FeaturesListOrder = [];
if nargin <= 2
    EliminationQuantity = FeatureQuantity;
end
while length(FeaturesListOrder) < FeatureQuantity
    RFE_Quantity = RFE_Quantity + 1;
    disp(['The ' num2str(RFE_Quantity) 'th SVM!']);
    
    if nargin <= 2
        EliminationQuantity = round(EliminationQuantity / 2);
    else
        if EliminationQuantity >= 10000 & FeatureQuantity - length(FeaturesListOrder) <= 10000
            EliminationQuantity = 1000;
        end
        if EliminationQuantity >= 1000 & FeatureQuantity - length(FeaturesListOrder) <= 1000
            EliminationQuantity = 100;
        end
    end
    
    Subjects_Data_New = Subjects_Data(:, Feature_Index);
    w_Brain = W_Calculate_SVM(Subjects_Data_New, Subjects_Label, 'Normalize');
    
    [~, Origin_Index] = sort(abs(w_Brain), 2);
    DeletedIndex = Origin_Index(1:EliminationQuantity);
    
    FeaturesListOrder = [Feature_Index(DeletedIndex) FeaturesListOrder]; % From important to non-important
    
    Subjects_Data_New(:, DeletedIndex) = [];
    Feature_Index(:, DeletedIndex) = [];
end

