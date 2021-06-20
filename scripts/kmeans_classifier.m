clc, clear
% Implementación del kmeans
% 1) Dividir cada clase de dígitos en 3 subclases
% 2) Se entrena el algoritmo de clasificación con las 30 clases
% 3) Se vuelve a pasar de las 30 clases a las 10 clases 

% Se cargan los datos del dataset de los digitos
load('../data/Trainnumbers.mat');

% Se normalizan los datos
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
    image_n(:,i)=(Trainnumbers.image(:,i)-mean_image)./std_image;
end

% Se dividen los datos en vectores correspondientesa a cada clase
class_0 = image_n(:,(Trainnumbers.label==0));
class_1 = image_n(:,(Trainnumbers.label==1));
class_2 = image_n(:,(Trainnumbers.label==2));
class_3 = image_n(:,(Trainnumbers.label==3));
class_4 = image_n(:,(Trainnumbers.label==4));
class_5 = image_n(:,(Trainnumbers.label==5));
class_6 = image_n(:,(Trainnumbers.label==6));
class_7 = image_n(:,(Trainnumbers.label==7));
class_8 = image_n(:,(Trainnumbers.label==8));
class_9 = image_n(:,(Trainnumbers.label==9));

% Se aplica el kmeans sobre cada una de las 10 clases
class0 = kmeans(class_0',3)';
class1 = kmeans(class_1',3)';
class2 = kmeans(class_2',3)';
class3 = kmeans(class_3',3)';
class4 = kmeans(class_4',3)';
class5 = kmeans(class_5',3)';
class6 = kmeans(class_6',3)';
class7 = kmeans(class_7',3)';
class8 = kmeans(class_8',3)';
class9 = kmeans(class_9',3)';

% Se asignas las subclases del 1 al 30 en orden creciente de clases
for i=1:1000
    class1(i) = class1(i) + 3;
    class2(i) = class2(i) + 6;
    class3(i) = class2(i) + 9;
    class4(i) = class2(i) + 12;
    class5(i) = class2(i) + 15;
    class6(i) = class2(i) + 18;
    class7(i) = class2(i) + 21;
    class8(i) = class2(i) + 24;
    class9(i) = class2(i) + 27;
end

% Se juntan todos los datos en una sola matriz de datos con 30 clases
% Matriz de datos y clases ordenadas
imagenes_norm = [class_0 class_1 class_2 class_3 class_4 class_5 class_6 class_7 class_8 class_9];
clases_imagenes_norm = [class0 class1 class2 class3 class4 class5 class6 class7 class8 class9];

% Colocación aleatoria de los datos de entrenamiento
Indexes = randperm(10000);
Training_Set.image = imagenes_norm(:,Indexes(1:8000));
Training_Set.label = clases_imagenes_norm(1,Indexes(1:8000));
Testing_Set.image = imagenes_norm(:,Indexes(8001:end));
Testing_Set.label = clases_imagenes_norm(1,Indexes(8001:end));

% Se aplica la PCA sobre los datos para reducir dimensionalidad
[image_trans, transMat] = processpca(Training_Set.image,0.0042);
test_pca = transMat.inverseTransform'*Testing_Set.image;

% Se entrena y testea el algoritmo
% k-nn classifier
mdl_knn = fitcknn(image_trans',Training_Set.label','NumNeighbors',3,'Standardize',1);
ans_test = predict(mdl_knn,test_pca');
num_errores_knn=length(find(ans_test'~=Testing_Set.label));
pred_rate_knn = (length(Testing_Set.label)-num_errores_knn)/length(Testing_Set.label);

% Se vuelve a pasar a las 10 clases las 2000 muestras de test
ans_test = ans_test';
for i=1:length(Testing_Set.label)
    if (ans_test(i) == 1 || ans_test(i) == 2 || ans_test(i) == 3)
        ans_test(i) = 0;
    elseif (ans_test(i) == 4 || ans_test(i) == 5 || ans_test(i) == 6)
        ans_test(i) = 1;
    elseif (ans_test(i) == 7 || ans_test(i) == 8 || ans_test(i) == 9)
        ans_test(i) = 2;
    elseif (ans_test(i) == 10 || ans_test(i) == 11 || ans_test(i) == 12)
        ans_test(i) = 3;
    elseif (ans_test(i) == 13 || ans_test(i) == 14 || ans_test(i) == 15)
        ans_test(i) = 4;
    elseif (ans_test(i) == 16 || ans_test(i) == 17 || ans_test(i) == 18)
        ans_test(i) = 5;
    elseif (ans_test(i) == 19 || ans_test(i) == 20 || ans_test(i) == 21)
        ans_test(i) = 6;
    elseif (ans_test(i) == 22 || ans_test(i) == 23 || ans_test(i) == 24)
        ans_test(i) = 7;
    elseif (ans_test(i) == 25 || ans_test(i) == 26 || ans_test(i) == 27)
        ans_test(i) = 8;
    else
        ans_test(i) = 9;
    end
end

