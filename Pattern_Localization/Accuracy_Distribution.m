function Accuracy_Distribution(AccuracyFileCell, ResultantFolder)

for i = 1:length(AccuracyFileCell)
    tmp = load(AccuracyFileCell{i});
    Accuracy(i) = tmp.Accuracy;
end
figure;
hist(Accuracy);
title('Accuracy distribution');
saveas(gcf, [ResultantFolder filesep 'Accuracy_distribution'], 'fig');
disp('max accuracy = ');
disp(max(Accuracy));