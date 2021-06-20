%% Linear Fishcer Disciminant for reducing the dimensionality
% Load the data
rng('default')%this restat the seed for the random values
load ../data/Trainnumbers.mat
Indexes = randperm(10000);
Training_Set.image = Trainnumbers.image(:,Indexes(1:8000));
Training_Set.label = Trainnumbers.label(1,Indexes(1:8000));
Testing_Set.image = Trainnumbers.image(:,Indexes(8001:end));
Testing_Set.label = Trainnumbers.label(:,Indexes(8001:end));
% Normalization
[image_n,ps0] = mapstd(Training_Set.image);
means = mean(image_n')';
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
% W = 0;
% B = 0;
for i=1:10
    %Sin
    Sin{i} = cov(classes{i}(:,:)',1)*classes_NUM(i);
    SWn = SWn + Sin{i};
    mci = mean(classes{i}(:,:)')';
    SBn = SBn + classes_NUM(i)*(mci - means)*(mci - means)';
%     xi=classes{i}(:,:)-repmat(mci,[1 classes_NUM(i)]); %Xi-MeanXi
%     W=W+xi*xi';                         %CALCULATE W
%     B=B+classes_NUM(i)*(mci-means)'*(mci-means); %CALCULATE B
end
% %W = inv(W)*B;
% 
%clean of the SWn. There are many features that do not have relation with
%the other features.
% SWn_clear = zeros(784,784);
% for i=1:784
%     for j=1:784
%         if SWn(i,j) == 0
%             SWn(i,j) = 0.01;
%         end
%     end
% end 
% W = inv(SWn)*SBn;
SWn_clear = SWn;
SBn_clear = SBn;
clean=0;
for i=1:784
    if sum(SWn(:,i))==0
            SWn_clear(:,i-clean) = [];   
            SWn_clear(i-clean,:) = []; 
            SBn_clear(:,i-clean) = [];   
            SBn_clear(i-clean,:) = []; 
            clean = clean + 1;
    end
end
W = inv(SWn_clear)*SBn_clear;
[W,Diagn]=eig(W);
%It is necessary to choose 10 -1 classes.
selection = sum(Diagn);
B = sort(selection,'descend');
B = B(1:10);
%It is necessary to choose 10 -1 classes.
selection = sum(Diagn);
D2 = sort(selection,'descend');
D2 = D2(1:2);


% SWn = SWn_clear;
% clean=784;
% insert=1;
% for i=1:784
%     if sum(SWn(i,:))==0
%             SWn_clear(insert, :) = SWn(i,:);
%             SWn_clear(clean, :) = [];
%             insert = insert + 1;
%             clean = clean - 1;
%     end
% end
% SWn = SWn_clear;
% S1 = cov(classes{1}(:,:)',1)*classes_NUM(1);
% SWn = inv(SWn'*SWn)*SWn';
% W = inv(W'*W)*W';
% Wn=inv(SWn)*(SBn);
% [image_n,ps0] = mapstd(Training_Set.image);
% [test_n,ps1] = mapstd(Testing_Set.image);
% 
% S0n = cov(class_0',1)*classes_NUM(1);
% S1n = cov(class_1',1)*classes_NUM(2);
% S2n = cov(class_2',1)*classes_NUM(3);
% S3n = cov(class_3',1)*classes_NUM(4);
% S4n = cov(class_4',1)*classes_NUM(5);
% S5n = cov(class_5',1)*classes_NUM(6);
% S6n = cov(class_6',1)*classes_NUM(7);
% S7n = cov(class_7',1)*classes_NUM(8);
% S8n = cov(class_8',1)*classes_NUM(9);
% S9n = cov(class_9',1)*classes_NUM(10);
% SWn = S0n +S1n + S2n + S3n +S4n + S5n +S6n +S7n + S8n +S9n;

%% Feature separability