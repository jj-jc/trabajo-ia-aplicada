
% MLP Classifier
clc, clear
% Load the data
load ../data/Trainnumbers.mat

% Changing the label matrix in order to work with 10 neurons in the output layer
digits_label = zeros(10,10000);
for i=1:10000
    digits_label(Trainnumbers.label(1,i)+1,i)= 1;
end

Indexes = randperm(10000);
Training_Set.image = Trainnumbers.image(:,Indexes(1:9000));
Training_Set.label = digits_label(:,Indexes(1:9000));
Testing_Set.image = Trainnumbers.image(:,Indexes(9001:end));
Testing_Set.label = digits_label(:,Indexes(9001:end));

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

%Normalization of the Testing Set (Same functionality)
[test_n,ps1] = mapstd(Testing_Set.image);

% Reduction of the dimension of the characteristics with PCA method
[image_trans, transMat] = processpca(image_n,0.0045);
test_pca = transMat.inverseTransform'*test_n;

% MLP Classifier

% MLP BUILDING 
net = feedforwardnet(10);
%net = newff(image_trans,Training_Set.label,100);
net.trainParam.epochs=100;
%net.trainFcn = 'trainlm';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'tansig';

% MLP TRAINING
[net,tr]=train(net,image_trans,Training_Set.label);

% ANSWER
ans_test=net(test_pca);
ans_test_normalized = vec2ind(ans_test);
ans_train=net(image_trans);
ans_train_normalized = vec2ind(ans_train);

% COMPUTE ERRORS
num_errors_test=length(find(ans_test_normalized~=vec2ind(Testing_Set.label)));
num_errors_train=length(find(ans_train_normalized~=vec2ind(Training_Set.label)));

% CONFUSION MATRIX
plotconfusion(Testing_Set.label,ans_test)

% PERFORMANCE
performance_test = (2000-num_errors_test)/2000;
performance_train = (8000-num_errors_train)/8000;
[c,cm]=confusion(Testing_Set.label,ans_test);





