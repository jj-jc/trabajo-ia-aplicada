function [digit] = imagen(image)
    for i=1:28
        for j=1:28
            digit(i,j)=image((i-1)*28+j);
        end
    end
end