%8.0395 accuracy
categories={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'};
imds = imageDatastore('F:\P3\Project\IndianCulturalEventRecognition\','includeSubfolders',true,'LabelSource','foldernames');
g=length(imds.Files);
[trainingset,testset]=splitEachLabel(imds,0.6);
training_features=[];
test_features=[];
traininglen=length(trainingset.Files);
testlen=length(testset.Files);
for k=1:traininglen
    im=imread(imds.Files{k});
    h=HOG(im);
   training_features=[training_features,h];
end
for k=1:testlen
    im=imread(imds.Files{k});
    h=HOG(im);
   test_features=[test_features,h];
end
training_label =trainingset.Labels;
test_label=testset.Labels;
sv=fitcknn(training_features',training_label,'NumNeighbors',5);
out=predict(sv,test_features');
count=0;
len=length(out);
for k=1:len
    if(out(k)==test_label(k))
        count=count+1;
    end
end
accuracy=(count/len)*100;
disp(accuracy);