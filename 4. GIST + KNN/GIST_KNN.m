% 22.3982 percent accuracy
close all
categories={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'};
imds = imageDatastore('F:\P3\Project\IndianCulturalEventRecognition\','includeSubfolders',true,'LabelSource','foldernames');
%imds=imageDatastore('C:\MY DATA\COLLEGE DATA\P cube\project\dataset\monument classified architecture','includeSubfolders',true,'labelSource','foldernames');
[training_set,testing_set]=splitEachLabel(imds,0.75);


%%
g=length(training_set.Files);
training_features=[];
for k=1:g
    k
    im1=imread(training_set.Files{k});
    clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist requires 1) prefilter image, 2) filter image and collect
% output energies
[gist1, param] = LMgist(im1, '', param);

    training_features=[training_features;gist1];
end
%%
g=length(testing_set.Files);
testing_features=[];
for l=1:g
    l
    im1=imread(testing_set.Files{l});
    % Parameters:
clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist requires 1) prefilter image, 2) filter image and collect
% output energies
[gist1, param] = LMgist(im1, '', param);

    testing_features=[testing_features;gist1];
end
%%
training_label=training_set.Labels;
test_label=testing_set.Labels;
sv=fitcknn(training_features,training_label,'NumNeighbors',5);
out=predict(sv,testing_features);
%%
r=out==test_label;
l=length(test_label);
count=0;
disp('accuracy =');

for m=1:l
    if(r(m)==1)
        count=count+1;
    end
end
a=count*100;
a=a/l;
disp(a);
%accu=accuracy(test_label,out);
%disp('accuracy=');
%disp(accu);