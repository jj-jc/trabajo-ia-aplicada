%% Linear Fishcer Disciminant for reducing the dimensionality
% Load the data
load ../data/Trainnumbers.mat
Indexes = randperm(10000);
Training_Set.image = Trainnumbers.image(:,Indexes(1:8000));
Training_Set.label = Trainnumbers.label(1,Indexes(1:8000));
Testing_Set.image = Trainnumbers.image(:,Indexes(8001:end));
Testing_Set.label = Trainnumbers.label(:,Indexes(8001:end));
% Normalization
[image_n,ps0] = mapstd(Training_Set.image);
image_n = Training_Set.image;
means = sum(image_n')';

% Divide the images in their corresponding class
class_0 = image_n(:,(Training_Set.label==0));
class_1 = image_n(:,(Training_Set.label==1));
class_2 = image_n(:,(Training_Set.label==2));
class_3 = image_n(:,(Training_Set.label==3));
class_4 = image_n(:,(Training_Set.label==4));
class_5 = image_n(:,(Training_Set.label==5));
class_6 = image_n(:,(Training_Set.label==6));
class_7 = image_n(:,(Training_Set.label==7));
class_8 = image_n(:,(Training_Set.label==8));
class_9 = image_n(:,(Training_Set.label==9));

classes = {class_0, class_1, class_2, class_3, ...
    class_4, class_5, class_6, class_7, class_8, class_9};
classes_NUM = [size(class_0,2), size(class_1,2), size(class_2,2), ...
    size(class_3,2), size(class_4,2), size(class_5,2), size(class_6,2),...
    size(class_7,2), size(class_8,2), size(class_9,2)];

SWn = 0;
SBn = 0;
for i=1:10
    %Sin
    Sin{i} = cov(classes{i}(:,:)',1)*classes_NUM(i);
    SWn = SWn + Sin{i};
    mean_class = mean(classes{i}(:,:)')';
    SBn = SBn + classes_NUM(i)*(mean_class - means)*(mean_class - means)';
end
SWn = inv(SWn'*SWn)*SWn';
Wn=inv(SWn)*(SBn);
% [image_n,ps0] = mapstd(Training_Set.image);
% [test_n,ps1] = mapstd(Testing_Set.image);

%% Feature separability

