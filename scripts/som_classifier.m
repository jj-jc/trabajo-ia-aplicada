
% SOM Classifier
%0.045 PCA, 800 epoch, dim 15 : 85% pred rate
%0.045 PCA, 800 epoch, dim 20 : 88% pred rate
%0.045 PCA, 800 epoch, dim 30 : 91.7% pred rate
%clc, clear

% Load the data
load('SOM_net.mat')
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

%Normalization of the Testing Set (Same functionality
[test_n,ps1] = mapstd(Testing_Set.image);
% Reduction of the dimension of the characteristics with PCA method
[image_trans, transMat] = processpca(image_n,0.0045);
%[image_trans, transMat] = processpca(Trainnumbers.image,0.001); no normalized
test_pca = transMat.inverseTransform'*test_n;


% Reconstruction of the images
anspcan=transMat.transform'*image_trans;

% Denormalization
for i=1:N
    anspca(:,i)=anspcan(:,i).*std_image+mean_image;
end

% Classifier

dim = 30;
net = selforgmap([dim dim]);
net.trainParam.epochs=800;
net = train(net,image_trans);

%view(net)
y_train = net(image_trans);
classes = vec2ind(y_train);
y_test = net(test_pca);
classes_test = vec2ind(y_test);
%Matriz que relaciona las clases del SOM con las clases reales de las
%muestras
SOM_Classes = zeros(dim*dim,10);
for j=1:N
    SOM_Classes(classes(j),Training_Set.label(j)+1) = SOM_Classes(classes(j),Training_Set.label(j)+1)+1;
end

%Matriz de correspondencia matriz SOM con clase predicha
%Aquí se asigna a cada clase SOM un número en función de los números más
%repetidos para cada una de ellas
for i=1:length(SOM_Classes)
    [mx, in] = max(SOM_Classes(i,:));
    SOM_Matrix(i) = in-1;
end

%Ahora se tienen que sacar los labels según el SOM obtenidos en la variable
%classes

for i=1:N
    SOM_pred(i) = SOM_Matrix(classes(i));
end

num_errores_SOM = length(find(SOM_pred~=Training_Set.label));
pred_rate_SOM = (length(Training_Set.label)-num_errores_SOM)/length(Training_Set.label);

for i=1:10000-N
    SOM_pred_test(i) = SOM_Matrix(classes_test(i));
end
num_errores_test_SOM = length(find(SOM_pred_test~=Testing_Set.label));
pred_rate_SOM_Test = (length(Testing_Set.label)-num_errores_test_SOM)/length(Testing_Set.label);

% Confusion matrix
C = confusionmat(SOM_pred', Testing_Set.label);
confusionchart(C);