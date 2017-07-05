function [Splited_Data Splited_Data_Label Origin_ID] = Split_NFolds(Subjects_Data, Subjects_Label, FoldQuantity)

if sum(ismember(Subjects_Label, -1))
    Group0_Flag = -1;
else
    Group0_Flag = 0;
end

Group1_Index = find(Subjects_Label == 1);
Group0_Index = find(Subjects_Label == Group0_Flag);
Group1_Index = Group1_Index';
Group0_Index = Group0_Index';
Group1_Data = Subjects_Data(Group1_Index, :);
Group0_Data = Subjects_Data(Group0_Index, :);
Group1_Label = Subjects_Label(Group1_Index);
Group0_Label = Subjects_Label(Group0_Index);
Group1_Quantity = length(Group1_Index);
Group0_Quantity = length(Group0_Index);

% Split group1
Group1_EachPart_Quantity = fix(Group1_Quantity / FoldQuantity);
Group1_RandID = randperm(Group1_Quantity);
for i = 1:FoldQuantity
    Splited_Data{i} = [];
    Splited_Data_Label{i} = [];
    Origin_ID{i} = [];
    if i == 1
        Splited_Data{i} = [Splited_Data{i} ; Group1_Data(Group1_RandID([1 : i * Group1_EachPart_Quantity]), :)];
        Origin_ID{i} = [Origin_ID{i} ; Group1_Index(Group1_RandID([1 : i * Group1_EachPart_Quantity]))];
    else
        Splited_Data{i} = [Splited_Data{i} ; Group1_Data(Group1_RandID([(i - 1) * Group1_EachPart_Quantity + 1: i * Group1_EachPart_Quantity]), :)];
        Origin_ID{i} = [Origin_ID{i} ; Group1_Index(Group1_RandID([(i - 1) * Group1_EachPart_Quantity + 1: i * Group1_EachPart_Quantity]))]; 
    end
    Splited_Data_Label{i} = [Splited_Data_Label{i} ; ones(Group1_EachPart_Quantity, 1)];
end
Group1_Reamin = mod(Group1_Quantity, FoldQuantity);
for i = 1:Group1_Reamin
    Splited_Data{i} = [Splited_Data{i} ; Group1_Data(Group1_RandID(FoldQuantity * Group1_EachPart_Quantity + i), :)];
    Origin_ID{i} = [Origin_ID{i} ; Group1_Index(Group1_RandID(FoldQuantity * Group1_EachPart_Quantity + i))]; 
    Splited_Data_Label{i} = [Splited_Data_Label{i} ; 1];
end

% Split group0
Group0_EachPart_Quantity = fix(Group0_Quantity / FoldQuantity);
Group0_RandID = randperm(Group0_Quantity);
for i = 1:FoldQuantity
    if i == 1
        Splited_Data{i} = [Splited_Data{i} ; Group0_Data(Group0_RandID([1 : i * Group0_EachPart_Quantity]), :)];
        Origin_ID{i} = [Origin_ID{i} ; Group0_Index(Group0_RandID([1 : i * Group0_EachPart_Quantity]))]; 
    else
        Splited_Data{i} = [Splited_Data{i} ; Group0_Data(Group0_RandID([(i - 1) * Group0_EachPart_Quantity + 1: i * Group0_EachPart_Quantity]), :)];
        Origin_ID{i} = [Origin_ID{i} ; Group0_Index(Group0_RandID([(i - 1) * Group0_EachPart_Quantity + 1: i * Group0_EachPart_Quantity]))];
    end
    Splited_Data_Label{i} = [Splited_Data_Label{i} ; Group0_Flag * ones(Group0_EachPart_Quantity, 1)];
end
Group0_Reamin = mod(Group0_Quantity, FoldQuantity);
for i = 1:Group0_Reamin
    Splited_Data{i} = [Splited_Data{i} ; Group0_Data(Group0_RandID(FoldQuantity * Group0_EachPart_Quantity + i), :)];
    Origin_ID{i} = [Origin_ID{i} ; Group0_Index(Group0_RandID(FoldQuantity * Group0_EachPart_Quantity + i))];
    Splited_Data_Label{i} = [Splited_Data_Label{i} ; Group0_Flag];
end

