
function PatternLocalization_MainFunction(Origin_Accuracy_Path, Rand_Accuracy_Path_Cell, Origin_w_Brain, Rand_w_Brain_Cell)

% Origin_Accuracy_Path:
%     The path of a .mat file, which contains a variable named 'Accuracy' representing the prediction accuracy.
%
% Rand_Accuracy_Path_Cell:
%     n*1 cell
%     Each element is a path of .mat file, which contains a variable named 'Accuracy' representing of the prediction accuracy of random sample (permutation test)
%
% Origin_w_Brain:
%     path of .mat file
%     a variable named 'w_Brain' should be stored in the mat file
%
% Rand_w_Brain_Cell:
%     cell of paths of .mat files
%     a variable named 'w_Brain' should be stored in each mat file
%

[ResultantFolder y z] = fileparts(Origin_Accuracy_Path);

disp('display accuracy distribution in random state:');
Accuracy_Distribution(Rand_Accuracy_Path_Cell, ResultantFolder);

disp('Claculate accuracy siginificance:');
Accuracy_Significance_Acc(Origin_Accuracy_Path, Rand_Accuracy_Path_Cell);

disp('display w distribution in random state:');
W_Distribution(Rand_w_Brain_Cell, 50, ResultantFolder);

disp('Permutation test for w Brain:');
Permutation_test(Origin_w_Brain, Rand_w_Brain_Cell, 0.05);
