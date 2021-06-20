clc; clear all; close all;
load('imagefileA.mat')
data = SOM_vis_Mat;
%  For Visualization of the input images 
data = double(data)';
for l = 1:10
    subplot(2,5,l);
    imshow(reshape(data(l,:),30,30));
end
data = double(data)';
drawnow
[~,N] = size(data); % Total input images
tot_Nodes = 100; % Total Nodes
w = rand(size(data,1), tot_Nodes); % Initial weights
eta_init = 0.1; % Initial learning rate
eta_updated = eta_init; % Variable learning rate
tot_epochs = 2000; % Total epochs
nhSize_init = 400; % Initial neighborhood size
nhSize_updated = nhSize_init; % Variable neighborhood size 
[I,J] = ind2sub([10, 10], 1:100); % Mapping from linear indexes to subscripts
tic;
for i=1:tot_epochs
%     epochsinfo = ['Epoch# ' , int2str(i)];
%     disp(epochsinfo);
    for j=1:N
        x = data(:,j);
%       Step 1: Calculating the Best matching unit
        dInput = sum( sqrt((w - repmat(x,1,tot_Nodes)).^2),1); % Distance from the input
        [~,ind] = min(dInput);
        bmu = [I(ind), J(ind)];
 
%       Step 2: Applying Gaussian function on the distance between the winner and other nodes
        dInput = 1/(sqrt(2*pi)*nhSize_updated).*exp(sum(( ([I( : ), J( : )]-repmat(bmu, tot_Nodes,1)).^2),2)/(-2*nhSize_updated));
        
%       Step 3: Weight update rule
        for s = 1:100
            w(:,s) = w(:,s) + eta_updated.*dInput(s).*( x - w(:,s));
        end
    end
 
%   Step 4: Updating the learning rate and neighborhood size
    eta_updated = eta_init * exp(-i/tot_epochs);
    nhSize_updated = nhSize_init*exp(-i/(1000/log(nhSize_init)));
    
    err_curve(i) = eta_updated;
    sig_curve(i) = nhSize_updated;
%   Display of Weights after every 200 epochs
    if mod(i,200) == 1
        figure;
        for l = 1:100
            subplot(10,10,l);
            imshow(reshape(w(:,l),30,30));
        end
%         drawnow
    end
end
toc;
% Curve plots
figure
title('Curve Plots');
subplot(2,1,1);plot(1:i, err_curve,'r')
xlabel('Epochs');ylabel('Learning Rate');
subplot(2,1,2); plot(1:i, sig_curve,'b')
xlabel('Epochs');ylabel('Neighboorhood Size');