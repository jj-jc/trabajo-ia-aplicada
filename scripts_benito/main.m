clc, clear

% Load the data
load ../data/Trainnumbers.mat
Indexes = randperm(10000);
Training_Set.image = Trainnumbers.image(:,Indexes(1:7000));
Training_Set.label = Trainnumbers.label(1,Indexes(1:7000));
Testing_Set.image = Trainnumbers.image(:,Indexes(7001:end));
Testing_Set.label = Trainnumbers.label(:,Indexes(7001:end));
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
[image_trans, transMat] = processpca(image_n,0.004);
%[image_trans, transMat] = processpca(Trainnumbers.image,0.001); no normalized
test_pca = transMat.inverseTransform'*test_n;


% Reconstruction of the images
anspcan=transMat.transform'*image_trans;

% Desnormalization
for i=1:N
    anspca(:,i)=anspcan(:,i).*std_image+mean_image;
end

% Comparison between the original and reconstructed images
imshow([imagen(Training_Set.image(:,1)),imagen(anspca(:,1))]);
figure;
imshow([imagen(image_n(:,1)),imagen(anspcan(:,1))]);

% k-nn classifier
mdl_knn = fitcknn(image_trans',Training_Set.label','NumNeighbors',3,'Standardize',1);
pred_knn = predict(mdl_knn,test_pca');
num_errores_knn=length(find(pred_knn'~=Testing_Set.label));

for i=0:9
    err_knn(i+1) = length(find((pred_knn'~=Testing_Set.label) & (Testing_Set.label == i)));
    err_knn_2(i+1) = length(find((pred_knn'~=Testing_Set.label) & (pred_knn' == i))) ;
end
pred_rate_knn = (length(Testing_Set.label)-num_errores_knn)/length(Testing_Set.label);

% Bayes classifier
mdl_bayes = fitcnb(image_trans',Training_Set.label');
pred_bayes = predict(mdl_bayes,test_pca');
num_errores_bayes=length(find(pred_bayes'~=Testing_Set.label));
for i=0:9
    err_bay(i+1) = length(find((pred_bayes'~=Testing_Set.label) & (Testing_Set.label == i)));
end
pred_rate_bayes = (length(Testing_Set.label)-num_errores_bayes)/length(Testing_Set.label);


