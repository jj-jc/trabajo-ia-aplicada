clc, clear

load('transMat_MLP_100-50neurons.mat');
load('MLP_100-50neurons.mat');
load('../resultados/Test_numbers_HW1.mat');
load ../data/Trainnumbers.mat
% Se crea una matriz 10x10000 para establecer la clase a la que pertenece
% cada dígito del dataset
digits_label = zeros(10,10000);
for i=1:10000
    digits_label(Trainnumbers.label(1,i)+1,i)= 1;
end

% Se normalizan los datos con los que se va a testear
[test_n,ps1] = mapstd(Trainnumbers.image);

% Se reduce la dimensionalidad de los datos de test normalizados
test_pca = transMat.inverseTransform'*test_n;

% Se obtiene la clasificación de los datos con la red neuronal
tic
ans_test=net(test_pca);
toc
% Se pasa a una matriz 1x10000 los resultados de la clasificacion
ans_test_normalized = vec2ind(ans_test);
resultados_clasificacion = ans_test_normalized - ones(1,10000);

% Se calculan errores
num_errors_test=length(find(ans_test_normalized~=vec2ind(digits_label)));
num_errors_test2=length(find(resultados_clasificacion~=Trainnumbers.label));

% Se pinta matriz de confusion
plotconfusion(digits_label,ans_test)

% Se saca la gráfica de rendimiento
performance_test = (10000-num_errors_test)/10000;
