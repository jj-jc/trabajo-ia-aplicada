
% clc, clear, close all

%% Load the data
load ../data/Trainnumbers.mat
Indexes = randperm(10000);
Training_Set.image = Trainnumbers.image(:,Indexes(1:8000));
Training_Set.label = Trainnumbers.label(1,Indexes(1:8000));
Testing_Set.image = Trainnumbers.image(:,Indexes(8001:end));
Testing_Set.label = Trainnumbers.label(:,Indexes(8001:end));

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
% image_n = Training_Set.image;

% %Normalization of the Testing Set (Same functionality
% [test_n,ps1] = mapstd(Testing_Set.image);
% [xTrainImages,tTrain] = digitTrainCellArrayData;

%% Autocencoder
% Parameters
hiddensize = 150;
epochs = 400;% epoch = 200-400
rng('default')
% Training
% L2WeightRegularization: controls the impact of an L2 regularizer for the
% weights, is tupically quite small
% SparsityRegularization this enforce a constraint on the sparsity of the
% output from the hidden layer.
% SparsityProportion: it controls the sparsity of the output from the
% hidden layer. f SparsityProportion is set to 0.1, this is equivalent to 
% saying that each neuron in the hidden layer should have an average output of 0.1 over the training examples.
autoencoder = trainAutoencoder(image_n, hiddensize, ...
'EncoderTransferFunction', 'satlin', ...
'DecoderTransferFunction', 'purelin', ...
'MaxEpochs' , epochs, ...
'L2WeightRegularization',0.0001, ...
'SparsityRegularization',4,...
'SparsityProportion',0.1, ...
'ScaleData', false);

figure()
plotWeights(autoencoder);
% autoencoder = trainAutoencoder(xTrainImages)

% Getting output from autoencoder
feat = encode(autoencoder, image_n);
output = decode(autoencoder, feat);


% imageReconstructed = predict(autoencoder, image_n);
view(autoencoder)
figure()

plotWeights(autoencoder);
figure()
% for i = 1:9
%     subplot(3,3,i);
%     imshow([imagen(image_n(:,i))]);
% end


for i = 1:9
    figure()
    imshow([imagen(image_n(:,i)), imagen(output(:,i)), imagen(anspcan(:,i))]);
end
% Error calculation
mseError_autoencoder = mse(image_n-output);

% Reduction of the dimension of the characteristics with PCA method
[image_trans, transMat] = processpca(image_n,0.004);

% Reconstruction of the images
anspcan=transMat.transform'*image_trans;

figure;
imshow([imagen(image_n(:,1)),imagen(anspcan(:,1))]);
% Error calculation
mseError_pca = mse(image_n-anspcan);



