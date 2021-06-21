%Data comparison
clear;
load('M20037_knn.mat')
class_knn = class;
load('M20037_bay.mat')
class_bay = class;
load('M20037_som.mat')
class_som = class;

err_knn_bay = length(find(class_knn~=class_bay))/100;
err_knn_som = length(find(class_knn~=class_som))/100;
err_bay_som = length(find(class_bay~=class_som))/100;