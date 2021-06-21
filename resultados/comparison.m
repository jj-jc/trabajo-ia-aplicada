%Data comparison
clear;
load('M20037_knn.mat')
class_knn = class;
load('M20037_bay.mat')
class_bay = class;
load('M20037_som.mat')
class_som = class;
load('M20037_mlp.mat')
class_mlp = class;
load('M20037_soft.mat')
class_soft = class;
err_knn_bay = length(find(class_knn~=class_bay))/100;
err_knn_som = length(find(class_knn~=class_som))/100;
err_knn_mlp = length(find(class_knn~=class_mlp))/100;
err_knn_soft = length(find(class_knn~=class_soft))/100;
err_bay_som = length(find(class_bay~=class_som))/100;
err_bay_mlp = length(find(class_bay~=class_mlp))/100;
err_som_mlp = length(find(class_som~=class_mlp))/100;