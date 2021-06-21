
% Testing MLP algortihm
clc, clear

load('../scripts/transMat_MLP_100-50neurons.mat');
load('../scripts/MLP_100-50neurons.mat');
load('Test_numbers_HW1.mat');

% Se normalizan los datos con los que se va a testear
[test_n,ps1] = mapstd(Test_numbers.image);

% Se reduce la dimensionalidad de los datos de test normalizados
test_pca = transMat.inverseTransform'*test_n;

% Se obtiene la clasificación de los datos con la red neuronal
ans_test=net(test_pca);

% Se pasa a una matriz 1x10000 los resultados de la clasificacion
ans_test_normalized = vec2ind(ans_test);
resultados_clasificacion = ans_test_normalized - ones(1,10000);

M20037_mlp.name = {'German'; 'JuanJose'; 'Alvaro'};
M20037_mlp.PCA = 49;
M20037_mlp.class = resultados_clasificacion;
