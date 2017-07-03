categories = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'};
imds=imageDatastore('F:\P3\Project\IndianCulturalEventRecognition\','includeSubfolders',true,'LabelSource','foldernames');
g=length(imds.Files);
[trainingset,testset]=splitEachLabel(imds,0.6);
training_features=[];
test_features=[];
traininglen=length(trainingset.Files);
testlen=length(testset.Files);
for k=1:traininglen
    im=imread(imds.Files{k});
    param.imageSize = [256 256]; % it works also with non-square images
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    [gist1, param] = LMgist(im,'', param);
    training_features=[training_features;gist1];
end
for k=1:testlen
    im=imread(imds.Files{k});
    param.imageSize = [256 256]; % it works also with non-square images
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    [gist2, param] = LMgist(im,'', param);
    test_features=[test_features;gist2];
end
training_label =trainingset.Labels;
test_label=testset.Labels;
sv=fitcecoc(training_features',training_label,'Learners','Linear','Coding','onevsall','ObservationsIn','columns');
out=predict(sv,test_features);
count=0;
len=length(out);
for k=1:len
    if(out(k)==test_label(k))
        count=count+1;
    end
end
accuracy=(count/len)*100;
disp(accuracy);
%Accuracy 8.1805a