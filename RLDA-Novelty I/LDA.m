function [A3]=LDA(y2,Class_Index,Class,m)
for i=1:size(Class,2)
    idx=Class_Index.(Class(i));
    Class_data=y2(:,idx);
    Mean.(Class(i))=mean(Class_data,2);
    Nk.(Class(i))=size(Class_Index.(Class(i)),2);
end
Mean_Complete=mean(y2,2); %Mean Of Complete Dataset

%Calculation of Sb

d=size(Mean_Complete,1);
Sb=zeros(d,d);
for i=1:size(Class,2)
    Sb=Sb + Nk.(Class(i))*(Mean.(Class(i))-Mean_Complete)*((Mean.(Class(i))-Mean_Complete)') ;
end

%Claculation of Sw
Sw=zeros(d,d);
for i=1:size(Class,2)
    Dataset_Cls=y2(:,Class_Index.(Class(i)));
    %size(Dataset_Cls)
    
    for j=1:(Nk.(Class(i)))
        Sw= Sw + (Dataset_Cls(:,j)-Mean.(Class(i)))*(Dataset_Cls(:,j)-Mean.(Class(i)))' ;     
    end
end
Sw_inv=inv(Sw);
%Sw_inv=pinv(Sw)
Matrix=Sw_inv*Sb;
[V,D] = eig(Matrix);
[x,ind]=sort(diag(D),'descend');
V=V(:,ind);
A3=V(:,1:m)';

end