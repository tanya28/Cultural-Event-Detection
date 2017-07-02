categories={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'};
imds = imageDatastore('F:\P3\Project\IndianCulturalEventRecognition\','includeSubfolders',true,'LabelSource','foldernames');
g=length(imds.Files);
[trainingSet,testSet]=splitEachLabel(imds,0.6)
training_features=[];
test_features=[];
train_length=length(trainingSet.Files);
test_length=length(testSet.Files);
for k=1:train_length
    im=imread(trainingSet.Files{k});
    h=HOG(im);
    training_features=[training_features,h];
end
for k=1:test_length
    im=imread(testSet.Files{k});
    h=HOG(im);
    test_features=[test_features,h];
end
training_label =trainingSet.Labels;
testing_label =testSet.Labels;
sv=fitcecoc(training_features,training_label,'Learners','Linear','Coding','onevsall','ObservationsIn','columns');
out=predict(sv,test_features');

%%
count=0;
l=length(out);
for k=1:l
    if(out(k)==testing_label(k))
        count=count+1;
    end
end
accuracy=(count/(length(testing_label)))*100;
disp(accuracy);
%%
%Accuracy 24.2595


