% 22.4490 percent accuracy
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
    im=rgb2gray(im1);
    h2=extractLBPFeatures(im);
    training_features=[training_features;h2];
end
%%
g=length(testing_set.Files);
testing_features=[];
for l=1:g
    l
    im1=imread(testing_set.Files{l});
    im=rgb2gray(im1);
    testing_features=[testing_features;extractLBPFeatures(im)];
end
%%

training_label=training_set.Labels;
test_label=testing_set.Labels;
sv=fitensemble(training_features,training_label, 'Bag',100,'Tree','Type', 'classification');
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