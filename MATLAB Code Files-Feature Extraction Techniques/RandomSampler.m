function [x_train,y_train,x_test,y_test,org_index_T,org_index_V]=RandomSampler(Dataset_mat,GroundTruth_mat,Class_Index,Class,n)
    x_t=[];
    y_t=[];
    x_val=[];
    y_val=[];
    org_index_T=[];
    org_index_V=[];
    for i=1:size(Class,2)
        Class_I=Class_Index.(Class(i));
        Total_samples=size(Class_I,2);
        Train_index=randperm(Total_samples,n);
        org_index_T=[org_index_T,Train_index];
            for i=1:size(Train_index,2)
                col=Class_I(1,Train_index(i));
                x_t=[x_t,Dataset_mat(:,col)];   
                y_t=[y_t,GroundTruth_mat(:,col)];
            end
        Class_I(:,Train_index)=[];
        org_index_V=[org_index_V,Class_I];
            for i=1:size(Class_I,2)
                col=Class_I(:,i);
                x_val=[x_val,Dataset_mat(:,col)];
                y_val=[y_val,GroundTruth_mat(:,col)];
            end
    end
    idx1=randperm(size(x_t,2),size(x_t,2));

    x_train=x_t(:,idx1);
    y_train=y_t(:,idx1);
    org_index_T=org_index_T(:,idx1);
    idx2=randperm(size(x_val,2),size(x_val,2));
    x_test=x_val(:,idx2);
    y_test=y_val(:,idx2);
    org_index_V=org_index_V(:,idx2);
end