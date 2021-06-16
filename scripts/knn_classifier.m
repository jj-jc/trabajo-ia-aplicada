
% K-NN CLASSIFIER (92% mean success rate)

clc, clear

% Load the data
load ../data/Trainnumbers.mat
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

%Normalization of the Testing Set (Same functionality)
[test_n,ps1] = mapstd(Testing_Set.image);

% Reduction of the dimension of the characteristics with PCA method
[image_trans, transMat] = processpca(image_n,0.0042);

%[image_trans, transMat] = processpca(Trainnumbers.image,0.001); no normalized
test_pca = transMat.inverseTransform'*test_n;

% test_pca2 = processpca(image_n,0.0042);transMat.inverseTransform'*test_n;

% Reconstruction of the images
anspcan=transMat.transform'*image_trans;
test_reconstructed=transMat.transform'*test_pca;

% Desnormalization
for i=1:N
    anspca(:,i)=anspcan(:,i).*std_image+mean_image;
end

% Comparison between the original and reconstructed images
%imshow([imagen(Training_Set.image(:,1)),imagen(anspca(:,1))]);
%figure;
%imshow([imagen(image_n(:,1)),imagen(anspcan(:,1))]);

% k-nn classifier
mdl_knn = fitcknn(image_trans',Training_Set.label','NumNeighbors',3,'Standardize',1);
pred_knn = predict(mdl_knn,test_pca');
num_errores_knn=length(find(pred_knn'~=Testing_Set.label));

for i=0:9
        for j=0:9
            err_knn(i+1,j+1) = length(find((pred_knn'~=Testing_Set.label) & (Testing_Set.label == i) & (pred_knn' == j)));
        end
end
    
MSE_training = immse(anspcan, image_n);
MSE_test = immse(test_reconstructed, test_n);
pred_rate_knn = (length(Testing_Set.label)-num_errores_knn)/length(Testing_Set.label);

% Confusion matrix
C = confusionmat(pred_knn', Testing_Set.label);
confusionchart(C);
