
% Bayesian Classifier (75% mean success rate)

clc, clear

% Load the data
load ../data/Trainnumbers.mat
load ../data/Test_numbers_HW1.mat
Indexes = randperm(10000);
Training_Set.image = Trainnumbers.image(:,Indexes(1:8000));
Training_Set.label = Trainnumbers.label(1,Indexes(1:8000));
Testing_Set.image = Trainnumbers.image(:,Indexes(8001:end));
Testing_Set.label = Trainnumbers.label(:,Indexes(8001:end));
% Divide the images in their corresponding class
class_0 = Training_Set.image(:,(Training_Set.label==0));
class_1 = Training_Set.image(:,(Training_Set.label==1));
class_2 = Training_Set.image(:,(Training_Set.label==2));
class_3 = Training_Set.image(:,(Training_Set.label==3));
class_4 = Training_Set.image(:,(Training_Set.label==4));
class_5 = Training_Set.image(:,(Training_Set.label==5));
class_6 = Training_Set.image(:,(Training_Set.label==6));
class_7 = Training_Set.image(:,(Training_Set.label==7));
class_8 = Training_Set.image(:,(Training_Set.label==8));
class_9 = Training_Set.image(:,(Training_Set.label==9));

% for i=1:4000
%      Training_Set.label(i) = Training_Set.label(i)*10;
% end
% Normalization of the learning data
[D,N] = size(Training_Set.image);
mean_image = mean(Training_Set.image')';
std_image = std(Training_Set.image')';
for j=1:D
    if std_image(j) == 0
        std_image(j) = 0.000001;
    end
end
image_n = zeros(D,N);
for i=1:N
    image_n(:,i)=(Training_Set.image(:,i)-mean_image)./std_image; % data normalized
end

%Normalization of the Testing Set (Same functionality
[test_n,ps1] = mapstd(Testing_Set.image);
% Reduction of the dimension of the characteristics with PCA method
[image_trans, transMat] = processpca(Training_Set.image,0.0045);
%[image_trans, transMat] = processpca(Trainnumbers.image,0.001); no normalized
test_pca = transMat.inverseTransform'*Testing_Set.image;


% Reconstruction of the images
anspcan=transMat.transform'*image_trans;

% Denormalization
for i=1:N
    anspca(:,i)=anspcan(:,i).*std_image+mean_image;
end

% Classifier
mdl_bayes = fitcnb(image_trans',Training_Set.label','OptimizeHyperparameters','auto');

pred_bayes = predict(mdl_bayes,test_pca');

num_errores_bayes=length(find(pred_bayes'~=Testing_Set.label));

pred_rate_bayes = (length(Testing_Set.label)-num_errores_bayes)/length(Testing_Set.label);

C = confusionmat(Testing_Set.label,pred_bayes);
cm = confusionchart(C);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

test_eval_bay = transMat.inverseTransform'*Test_numbers.image;
pred_test_bay = predict(mdl_bayes,test_eval_bay');