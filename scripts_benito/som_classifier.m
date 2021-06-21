
% SOM Classifier
%0.045 PCA, 800 epoch, dim 15 : 85% pred rate
%0.045 PCA, 800 epoch, dim 20 : 88% pred rate
%0.045 PCA, 800 epoch, dim 30 : 91.7% pred rate 90% test
%0.045 PCA, 1200 epoch, dim 35 : 92.5% pred rate (menor para test: 89%)

clear

% Load the data
%load('net.mat')
load ../data/Trainnumbers.mat
load('Colormap1.mat');
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
[image_trans, transMat] = processpca(Training_Set.image,0.0040);
%[image_trans, transMat] = processpca(Trainnumbers.image,0.001); no normalized
test_pca = transMat.inverseTransform'*Testing_Set.image;


% Reconstruction of the images
anspcan=transMat.transform'*image_trans;

% Denormalization
for i=1:N
    anspca(:,i)=anspcan(:,i).*std_image+mean_image;
end

% Classifier

dim = 25;
net = selforgmap([dim dim]);
net.trainParam.epochs=400;
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
    [mx, in] = max(SOM_Classes(i,:)); %Obtiene el maximo y el índice
    %Ahora hay que comprobar repeticiones del máximo y tratarlas
    max_indexes = find(SOM_Classes(i,:)==mx);
    %Si encuentra repetido el máximo, escoge el que tome el mismo valor
    %que el dato anterior (vecindad)
    %Si ninguno de los máximos coincide con el anterior, escoge el primero
    %que salga 
    if(length(max_indexes)>1 && i~=1)
        for j=1:length(max_indexes)
            if(mod(i,dim)~=1)
                if max_indexes(j)-1 == SOM_Matrix(i-1)
                    in = max_indexes(j);
                end
            else
                if max_indexes(j)-1 == SOM_Matrix(i-dim)
                    in = max_indexes(j);
                end
            end
        end
    end
    SOM_Matrix(i) = in-1;
    if (mx==0 && i~=1)
        if(mod(i,dim)~=1)
            SOM_Matrix(i) = SOM_Matrix(i-1);
        else
            SOM_Matrix(i) = SOM_Matrix(i-dim);
        end
    end
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
SOM_vis_Mat = transpose(reshape(SOM_Matrix,[dim,dim]));

hmo = HeatMap(flip(SOM_vis_Mat), 'Colormap', CustomColormap1);

hmo.Annotate = true;
view(hmo);

centers = net.IW{1};

centers_original_n = (transMat.transform'*centers');
for i=1:dim*dim
    centers_original{i}=imagen(centers_original_n(:,i));%.*std_image+mean_image);
end
figure;
montage(centers_original)