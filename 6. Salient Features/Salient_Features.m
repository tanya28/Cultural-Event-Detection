rootFolder = fullfile('F:\P3\Project\IndianCulturalEventRecognition');  
categories = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds);
convnet=helperImportMatConvNet('imagenet-caffe-alex.mat');
%%
disp('CNNhas been loaded');
%imds.ReadFcn = @(filename)readAndPreprocessImage_saliency(filename);
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
[trainingSet, testSet] = splitEachLabel(imds, 0.6, 'randomize');
featureLayer1 = 'fc7';
featureLayer2= 'fc6';
%%
 trainingFeatures1 = activations(convnet, trainingSet, featureLayer1, ...
     'MiniBatchSize', 32, 'OutputAs', 'columns');
 trainingFeatures2 = activations(convnet, trainingSet, featureLayer2, ...
     'MiniBatchSize', 31, 'OutputAs', 'columns');
% trainingFeatures3 = activations(convnet, trainingSet, featureLayer3, ...
 %    'MiniBatchSize', 31, 'OutputAs', 'columns');
trainingFeatures = [trainingFeatures1;trainingFeatures2];
trainingLabels = trainingSet.Labels;

%%
testFeatures1 = activations(convnet, testSet, featureLayer1, 'MiniBatchSize',32);
testFeatures2 = activations(convnet, testSet, featureLayer2, 'MiniBatchSize',32);
%testFeatures3 = activations(convnet, testSet, featureLayer3, 'MiniBatchSize',32);
%%
testFeature = [testFeatures1,testFeatures2];
classifier = fitcecoc(trainingFeatures,trainingLabels,'Learners','Linear','Coding','onevsall','ObservationsIn','columns');

predictedLabels = predict(classifier, testFeature);
testLabels = testSet.Labels;

confMat = confusionmat(testLabels, predictedLabels);
confMat2 = bsxfun(@rdivide,confMat,sum(confMat,2));
