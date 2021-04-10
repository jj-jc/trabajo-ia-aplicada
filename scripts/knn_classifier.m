
% K-NN CLASSIFIER

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

% Learning data (80%)
c0_l = class_0(:,1:800);
c1_l = class_1(:,1:800);
c2_l = class_2(:,1:800);
c3_l = class_3(:,1:800);
c4_l = class_4(:,1:800);
c5_l = class_5(:,1:800);
c6_l = class_6(:,1:800);
c7_l = class_7(:,1:800);
c8_l = class_8(:,1:800);
c9_l = class_9(:,1:800);
learning_data = [c0_l c1_l c2_l c3_l c4_l c5_l c6_l c7_l c8_l c9_l];

learning_class(1,1:800)=0;
learning_class(1,801:1600)=1;
learning_class(1,1601:2400)=2;
learning_class(1,2401:3200)=3;
learning_class(1,3201:4000)=4;
learning_class(1,4001:4800)=5;
learning_class(1,4801:5600)=6;
learning_class(1,5601:6400)=7;
learning_class(1,6401:7200)=8;
learning_class(1,7201:8000)=9;

% Testing data (20%)
c0_t = class_1(:,801:1000);
c1_t = class_2(:,801:1000);
c2_t = class_3(:,801:1000);
c3_t = class_4(:,801:1000);
c4_t = class_5(:,801:1000);
c5_t = class_6(:,801:1000);
c6_t = class_7(:,801:1000);
c7_t = class_8(:,801:1000);
c8_t = class_9(:,801:1000);
c9_t = class_9(:,801:1000);
testing_data = [c0_t c1_t c2_t c3_t c4_t c5_t c6_t c7_t c8_t c9_t];

testing_class(1,1:200)=0;
testing_class(1,201:400)=1;
testing_class(1,401:600)=2;
testing_class(1,601:800)=3;
testing_class(1,801:1000)=4;
testing_class(1,1001:1200)=5;
testing_class(1,1201:1400)=6;
testing_class(1,1401:1600)=7;
testing_class(1,1601:1800)=8;
testing_class(1,1801:2000)=9;

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

% Reduction of the dimension of the characteristics with PCA method
[learning_data_trans, transMat] = processpca(learning_data_n,0.001);
% learning_data_trans is a matrix with the learning images whose dimensionality has been reduced
% transMat is the transformation matrix


%[image_trans, transMat] = processpca(learning_data,0.001); no normalized

% Reconstruction of the learning images
anspcan=transMat.transform'*learning_data_trans;

% Desnormalization
for i=1:Nl
    anspca(:,i)=anspcan(:,i).*std_learning_data+mean_learning_data;
end

% Comparison between the original and reconstructed of learning images
imshow([imagen(learning_data(:,1000)),imagen(anspca(:,1000))]); % not normalized data
figure;
imshow([imagen(learning_data_n(:,1)),imagen(anspcan(:,1))]); % normalized data

% k-nn classifier
mdl_knn = fitcknn(anspca',learning_class','NumNeighbors',50,'Standardize',1);
pred_knn = predict(mdl_knn,testing_data');
num_errores_knn=length(find(pred_knn'~=testing_class));


