
% Testing k-nn algortihm
clc, clear

load('Test_numbers_HW1.mat');
load('../scripts/mdl_knn.mat');
load('../scripts/transMat_knn.mat');

% Se normalizan los datos con los que se va a testear
[test_n,ps1] = mapstd(Test_numbers.image);

% Se reduce la dimensionalidad de los datos de test normalizados
test_pca = transMat.inverseTransform'*test_n;

pred_knn = predict(mdl_knn,test_pca');

M20037_knn.name = {'German'; 'JuanJose'; 'Alvaro'};
M20037_knn.PCA = 52; % poner el número final
M20037_knn.class = pred_knn';