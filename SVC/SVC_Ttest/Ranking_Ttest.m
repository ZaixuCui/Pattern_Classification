function [PValues RetainID]= Ranking_Ttest(Subjects_Data, Subjects_Label, P_Threshold, Multiple_Correction)

%
% Subjects_Data:
%               matrix: m * n 
%               m is quantity of subjects; n is quantity of features
%
% Subjects_Label:
%               vector: m * 1; array of {1 or -1}
%               m is quantity of subjects
%
% P_Threshold:
%               0.05 or 0.01 is common
%
% Multiple_Correction:
%               structure with fields as listed:
%               
%               Flag:
%                   'Yes': do multiple comparison correction
%                   'No': don't do multiple comparison correction
%
%               Method:
%                   'FDR'
%
%               q:
%                   if the method is FDR, this field should be assigned a
%               value
%

%
% PValues:
%              p value for each feature acquired with two-sample t-test
%
% RetainID:
%              The ID of elements retained with certain
%              FeatureRetainQuantity or certain threshold
%

[SubjectQuantity FeatureQuantity] = size(Subjects_Data);

Group1Data = Subjects_Data(find(Subjects_Label == 1), :);
Group0Data = Subjects_Data(find(Subjects_Label == -1), :);

[SignificantVoxel, PValues] = ttest2(Group1Data, Group0Data, P_Threshold, 'both');

if nargin >= 4
    if strcmp(Multiple_Correction.Flag, 'Yes')
        if strcmp(Multiple_Correction.Method, 'FDR')
            Sig_FDR_ID = FDR_Correction(PValues, Multiple_Correction.q);
            RetainID = Sig_FDR_ID;
        end
    elseif strcmp(Multiple_Correction.Flag, 'No')
        RetainID = find(SignificantVoxel == 1);
    else
        error('The Flag field of Multiple_Correction should be ''Yes'' of ''No''.');
    end
else
    RetainID = find(SignificantVoxel == 1);
end
