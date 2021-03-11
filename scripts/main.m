clc, clear

% Load the data
load ../data/Trainnumbers.mat

% Divide the images in their corresponding class
class_0 = Trainnumbers.image(:,(Trainnumbers.label==0));
class_1 = Trainnumbers.image(:,(Trainnumbers.label==1));
class_2 = Trainnumbers.image(:,(Trainnumbers.label==2));
class_3 = Trainnumbers.image(:,(Trainnumbers.label==3));
class_4 = Trainnumbers.image(:,(Trainnumbers.label==4));
class_5 = Trainnumbers.image(:,(Trainnumbers.label==5));
class_6 = Trainnumbers.image(:,(Trainnumbers.label==6));
class_7 = Trainnumbers.image(:,(Trainnumbers.label==7));
class_8 = Trainnumbers.image(:,(Trainnumbers.label==8));
class_9 = Trainnumbers.image(:,(Trainnumbers.label==9));

% Normalization of the learning data
[D,N] = size(Trainnumbers.image);
mean_image = mean(Trainnumbers.image')';
std_image = std(Trainnumbers.image')';
for j=1:D
    if std_image(j) == 0
        std_image(j) = 0.000001;
    end
end
image_n = zeros(D,N);
for i=1:N
    image_n(:,i)=(Trainnumbers.image(:,i)-mean_image)./std_image; % data normalized
end

% Reduction the dimension of the characteristic with PCA method
[image_trans, transMat] = processpca(image_n,0.001);
%[image_trans, transMat] = processpca(Trainnumbers.image,0.001); no normalized

% Reconstruction of the images
anspcan=transMat.transform'*image_trans;

% Desnormalization
for i=1:N
    anspca(:,i)=anspcan(:,i).*std_image+mean_image;
end

% Comparison between the original and reconstructed images
imshow([imagen(Trainnumbers.image(:,1)),imagen(anspca(:,1))]);
figure;
imshow([imagen(image_n(:,1)),imagen(anspcan(:,1))]);

% k-nn classifier
% mdl_knn = fitcknn(image_trans',Trainnumbers.label','NumNeighbors',5,'Standardize',1);
% pred_knn = predict(mdl_knn,image_trans');
% num_errores_knn=length(find(pred_knn'~=Trainnumbers.label));

% Bayes classifier
mdl_bayes = fitcnb(image_trans',Trainnumbers.label');
pred_bayes = predict(mdl_bayes,image_trans');
num_errores_bayes=length(find(pred_bayes'~=Trainnumbers.label));




