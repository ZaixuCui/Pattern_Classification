function Ttest_SVM_2group_PSelection_SGE(Subjects_Data, Subjects_Label, P_Value_Range, Pre_Method, ResultantFolder, QueueOption)
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

save([ResultantFolder filesep 'SubjectData.mat'], 'Subjects_Data');
Subjects_Quantity = length(Subjects_Label);

for i = 1:Subjects_Quantity
    
    Job_Name1 = ['Folder_' num2str(i)];
    pipeline.(Job_Name1).command   = 'Ttest_SVM_2group_PSelection_SGE_child(opt.SubjectsData_Path, opt.Subjects_Label, opt.Folder_th, opt.P_Value_Range, opt.Pre_Method, opt.ResultantFolder)';
    pipeline.(Job_Name1).opt.SubjectsData_Path = [ResultantFolder filesep 'SubjectData.mat'];
    pipeline.(Job_Name1).opt.Subjects_Label = Subjects_Label;
    pipeline.(Job_Name1).opt.Folder_th = i;
    pipeline.(Job_Name1).opt.P_Value_Range = P_Value_Range;
    pipeline.(Job_Name1).opt.Pre_Method = Pre_Method;
    pipeline.(Job_Name1).opt.ResultantFolder = ResultantFolder;
    pipeline.(Job_Name1).files_out{1} = [ResultantFolder filesep 'predicted_labels_' num2str(i) '.mat'];
    pipeline.(Job_Name1).files_out{2} = [ResultantFolder filesep 'decision_values_' num2str(i) '.mat'];
    pipeline.(Job_Name1).files_out{3} = [ResultantFolder filesep 'RetainID_' num2str(i) '.mat'];
    pipeline.(Job_Name1).files_out{4} = [ResultantFolder filesep 'w_' num2str(i) '.mat'];
    
end

Job_Name2 = ['Results_Merge'];
for i = 1:Subjects_Quantity
    PreJob_Name = ['Folder_' num2str(i)];
    pipeline.(Job_Name2).command   = 'Ttest_SVM_2group_PSelection_SGE_child2(opt.ResultantFolder, opt.Subjects_Label)';
    pipeline.(Job_Name2).files_in{i} = pipeline.(PreJob_Name).files_out{4};
    pipeline.(Job_Name2).opt.ResultantFolder = ResultantFolder;
    pipeline.(Job_Name2).opt.Subjects_Label = Subjects_Label;
end

psom_gb_vars

Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = QueueOption;
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 1;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = [ResultantFolder filesep 'logs'];

psom_run_pipeline(pipeline,Pipeline_opt);
