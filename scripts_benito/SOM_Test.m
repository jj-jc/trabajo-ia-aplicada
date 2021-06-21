%SOM Test
clc, clear
load('SOM_35x35_9385.mat','SOM_Classes','net','Testing_Set','dim','N','classes_test','transMat');
load ../data/Trainnumbers.mat
load ../data/Test_numbers_HW1.mat
%load Trainnumbers.mat
load('Colormap1.mat');
for i=1:length(SOM_Classes)
    [mx, in] = max(SOM_Classes(i,:)); %Obtiene el maximo y el índice
    %Ahora hay que comprobar repeticiones del máximo y tratarlas
    max_indexes = find(SOM_Classes(i,:)==mx);
    %Si encuentra repetido el máximo, escoge el que tome el mismo valor
    %que el dato anterior (vecindad)
    %Si ninguno de los máximos coincide con el anterior, escoge el primero
    %que salga (lo suyo sería también comprobar el posterior
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

for i=1:10000-N
    SOM_pred_test(i) = SOM_Matrix(classes_test(i));
end
num_errores_test_SOM = length(find(SOM_pred_test~=Testing_Set.label));
pred_rate_SOM_Test = (length(Testing_Set.label)-num_errores_test_SOM)/length(Testing_Set.label);
SOM_vis_Mat = reshape(SOM_Matrix,[dim,dim]);

hmo = HeatMap(flip(SOM_vis_Mat'), 'Colormap', CustomColormap1);

hmo.Annotate = true;
%view(hmo);
centers = net.IW{1};

centers_original_n = (transMat.transform'*centers');
for i=1:dim*dim
    centers_original{i}=imagen(centers_original_n(:,i));%.*std_image+mean_image);
end
figure;
montage(centers_original)
figure;
C = confusionmat(SOM_pred_test,Testing_Set.label);
confusionchart(C);

test_eval_som = transMat.inverseTransform'*Test_numbers.image;
y_eval = net(test_eval_som);
classes_eval = vec2ind(y_eval);
for i=1:10000
    pred_test_SOM(i,1) = SOM_Matrix(classes_eval(i));
end