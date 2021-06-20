% 
% clc, clear, close all
% 
% %% Load the data
% rng('default')%this restat the seed for the random values, it 
% load ../data/Trainnumbers.mat
% Indexes = randperm(10000);
% Training_Set.image = Trainnumbers.image(:,Indexes(1:8000));
% Training_Set.label = Trainnumbers.label(1,Indexes(1:8000));
% Testing_Set.image = Trainnumbers.image(:,Indexes(8001:end));
% Testing_Set.label = Trainnumbers.label(:,Indexes(8001:end));
% 
% %Normalization
% [image_n,ps0] = mapstd(Training_Set.image);
% [test_n,ps1] = mapstd(Testing_Set.image);
% 
% %% ONLY PCA Dim=51
% % PCA
% [image_trans_, transMat] = processpca(image_n,0.0043);
% % Reconstruction of the images
% img_pca=transMat.transform'*image_trans_;
% % Error calculation
% mse_pca = mse(image_n-img_pca)
% 
% %% PCA AND AUTO-ENCODER
% % Reduction of the dimension of the characteristics with PCA method
% [image_trans, transMat] = processpca(image_n,0.002);
% % Reconstruction of the images
% anspcan=transMat.transform'*image_trans;
% 
% % Autocencoder
% 
% hiddensize = 51;
% epochs = 300;% epoch = 200-400
% % Training
% % L2WeightRegularization: controls the impact of an L2 regularizer for the
% % weights, is tupically quite small
% % SparsityRegularization this enforce a constraint on the sparsity of the
% % output from the hidden layer.
% % SparsityProportion: it controls the sparsity of the output from the
% % hidden layer. f SparsityProportion is set to 0.1, this is equivalent to 
% % saying that each neuron in the hidden layer should have an average output of 0.1 over the training examples.
% % 'EncoderTransferFunction', 'satlin', ...
% % 'DecoderTransferFunction', 'purelin', ...
% autoencoder = trainAutoencoder(image_n, hiddensize, ...
% 'EncoderTransferFunction', 'logsig', ...
% 'DecoderTransferFunction', 'purelin', ...
% 'MaxEpochs' , epochs, ...
% 'L2WeightRegularization',0.0001, ...
% 'SparsityRegularization',4,...
% 'SparsityProportion',0.4, ...
% 'ScaleData', false, ...
% 'UseGPU',true);
% % Getting output from autoencoder
% % feat = encode(autoencoder, image_n);
% % img_decode = decode(autoencoder, feat);
% img_decode = predict(autoencoder, image_n);
% % view(autoencoder)
% figure()
% % Error calculation
% mseError_autoencoder = mse(image_n-img_decode)
% 
% for i = 1:9
%     figure()
%     imshow([imagen(image_n(:,i)), imagen(img_decode(:,i)), imagen(img_pca(:,i))]);
% end

%% Classification with autoencoder and softmax (95.3%)
% %% Load the data
load ../data/Trainnumbers.mat
Indexes = randperm(10000);
Training_Set.image = Trainnumbers.image(:,Indexes(1:8000));
Training_Set.label = Trainnumbers.label(1,Indexes(1:8000));
Testing_Set.image = Trainnumbers.image(:,Indexes(8001:end));
Testing_Set.label = Trainnumbers.label(:,Indexes(8001:end));

[image_n,ps0] = mapstd(Training_Set.image);
[test_n,ps1] = mapstd(Testing_Set.image);
%first autoencoder
rng('default')
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(image_n,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
view(autoenc1)
figure()
plotWeights(autoenc1);
feat1 = encode(autoenc1,image_n);
%Second autoencoder
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
view(autoenc2)
feat2 = encode(autoenc2,feat1);

%Training softmax layer
number_classes=10;
dim_training = size(Training_Set.label,2);
training_labels = zeros(number_classes, dim_training);
for i=1:dim_training
    training_labels(Training_Set.label(i)+1,i) = 1;
end

number_classes=10;
dim_test = size(Testing_Set.label,2);
test_labels = zeros(number_classes, dim_test);
for i=1:dim_test
    test_labels(Testing_Set.label(i)+1,i) = 1;
end
softnet = trainSoftmaxLayer(feat2, training_labels,'MaxEpochs',400);
stackednet = stack(autoenc1,autoenc2,softnet);
view(stackednet)
y = stackednet(test_n);
plotconfusion(test_labels,y);
% backpropagation
stackednet = train(stackednet, image_n, training_labels);
y = stackednet(test_n);
plotconfusion(test_labels,y);