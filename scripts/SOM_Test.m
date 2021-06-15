%SOM Test
clc, clear
load('SOM_net.mat','net','transMat','SOM_Classes','SOM_Matrix');
load ../data/Trainnumbers.mat
[test_n,ps1] = mapstd(Trainnumbers.image);
test_pca = transMat.inverseTransform'*test_n;
y_test = net(test_pca);
classes_test = vec2ind(y_test);
for i=1:10000
    SOM_pred_test(i) = SOM_Matrix(classes_test(i));
end
num_errores_test_SOM = length(find(SOM_pred_test~=Trainnumbers.label));
pred_rate_SOM_Test = (length(Trainnumbers.label)-num_errores_test_SOM)/length(Trainnumbers.label);

SOM_vis_Mat = reshape(SOM_Matrix,[30,30]);
