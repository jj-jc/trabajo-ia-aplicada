
% SCRIPT DE PRUEBAS

% K-NN

% K-NN CLASSIFIER

clc, clear

% Load the data
load ../data/Trainnumbers.mat

% Learning data
learning_data = Trainnumbers.image(:,1:8000);
learning_class = Trainnumbers.label(:,1:8000);

% Testing data
testing_data = Trainnumbers.image(:,8001:10000);
testing_class = Trainnumbers.label(:,8001:10000);

% Normalization of the learning data
[Dl,Nl] = size(learning_data);
mean_learning_data = mean(learning_data')';
std_learning_data = std(learning_data')';
for j=1:Dl
    if std_learning_data(j) == 0
        std_learning_data(j) = 0.000001;
    end
end
learning_data_n = zeros(Dl,Nl);
for i=1:Nl
    learning_data_n(:,i)=(learning_data(:,i)-mean_learning_data)./std_learning_data; % learning data normalized
end

% Normalization of the testing data
[Dt,Nt] = size(testing_data);
mean_testing_data = mean(testing_data')';
std_testing_data = std(testing_data')';
for j=1:Dt
    if std_testing_data(j) == 0
        std_testing_data(j) = 0.000001;
    end
end
testing_data_n = zeros(Dt,Nt);
for i=1:Nt
    testing_data_n(:,i)=(testing_data(:,i)-mean_testing_data)./std_testing_data; % testing data normalized
end

% Reduction of the dimensionality using PCA
[transMatCL,DiagL] = eig(cov(learning_data_n'));
[transMatCT,DiagT] = eig(cov(testing_data_n'));
ncompca = 200;
for i=1:ncompca
    transMatL(i,:)=transMatCL(:,Dl+1-i)';
    transMatT(i,:)=transMatCT(:,Dl+1-i)';
end
learning_data_trans = transMatL*learning_data_n;
testing_data_trans = transMatT*testing_data_n;

% Reconstruction of the learning and testing images
learning_anspcan=transMatL'*learning_data_trans;
testing_anspcan=transMatT'*testing_data_trans;

% Desnormalization
for i=1:Nl
    learning_anspca(:,i)=learning_anspcan(:,i).*std_learning_data+mean_learning_data;
end
for i=1:Nt
    testing_anspca(:,i)=testing_anspcan(:,i).*std_testing_data+mean_testing_data;
end

% Comparison between the original and reconstructed of learning images
imshow([imagen(learning_data(:,1000)),imagen(learning_anspca(:,1000))]); % not normalized learning data
figure;
imshow([imagen(learning_data_n(:,1)),imagen(learning_anspcan(:,1))]); % normalized learning data

% k-nn classifier
mdl_knn = fitcknn(learning_data_trans',learning_class','NumNeighbors',50,'Standardize',1);
pred_knn = predict(mdl_knn,testing_data_trans');
num_errores_knn=length(find(pred_knn'~=testing_class));


