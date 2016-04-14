function W_Distribution(w_Brain_Cell, Feature_ID, ResultantFolder)

for i = 1:length(w_Brain_Cell)
    tmp = load(w_Brain_Cell{i});
    w_Component(i) = tmp.w_Brain(Feature_ID);
end
figure(1);
hist(w_Component);
title(['w ' num2str(Feature_ID)]);
saveas(gcf, [ResultantFolder filesep 'w_' num2str(Feature_ID)], 'fig');