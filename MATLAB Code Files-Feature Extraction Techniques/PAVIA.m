%%%Dataset Loading%%%
Dataset=load('PaviaU.mat');
GroundTruth=load('PaviaU_gt.mat');
Dataset_mat=Dataset.paviaU ;
GroundTruth_mat=GroundTruth.paviaU_gt;

%%%Dataset:[number of bands,number of samples]%%%
[row,col,dimension]=size(Dataset_mat);
n=row*col;
Dataset=reshape(Dataset_mat,n,dimension)';

%%%Ground Truth :[1,number of samples]%%%
Class=["Asphalt","Meadows","Gravel","Trees","PaintedSheets","BareSoil","Bitumen","SelfBLockingBricks","Shadows"];
[row1,col1]=size(GroundTruth_mat);
n1=row1*col1;
GroundTruth=reshape(GroundTruth_mat,n1,1)';

%%% Index generation of Clsses%%%
Class_Index_Complete =Compute_Classindex(GroundTruth,Class);

%%Normalization%%%
Dataset=normalize(Dataset,2);

%%%Number of Training Samples per class%%%
No_TrainingSamples_Class=200;

[x_train,y_train,x_test,y_test,org_index_T,org_index_V]=RandomSampler(Dataset,GroundTruth,Class_Index_Complete,Class,No_TrainingSamples_Class);
  
%lambda=[10^-4,10^-3,10^-2,0.1,1,10,10^2,10^3,10^4]; %%%RLDA%%%

%accuracy_lambda =[]
accuracy=[];

% for i=1:size(lambda,2)
%     l=lambda(i)
for m=1:8 %Number Features To Be Selected
        if rem(m,2)==0
            m1=m/2;
            m2=m/2;
        else
            m1=(m+1)/2;
            m2=m-m1;
        end
        Class_Index_Train=Compute_Classindex(y_train,Class); %Class Index from Training Data
        Class_Index_Test=Compute_Classindex(y_test,Class);%Class Index from Testing Data
        
       %%%PCA on Training Data %%%
       [coeff,score,latent] = pca(x_train') ;
        A1=coeff(:,1:m1)';        
        A2=coeff(:,m1+1:103)';
        
       %%%Transformation%%%
        y1=A1*x_train;
        y2=A2*x_train;
        
       %%%LDA on Small Power Components%%%
        A3=LDA(y2,Class_Index_Train,Class,m2); 
        
        %%%RLDA on Small Power Components%%%
        %A3=RLDA(y2,Class_Index_Train,Class,m2,l);
        
        %%%Transformation%%%
        y3=A3*y2;
        y=[y1;y3];
        
       %%%Testing Samples Feautre extraction%%% 
       y1_Test_PCDA=A1*x_test;%Tranformed Data
       y2_Test_PCDA=A2*x_test;% Tranformed Data
        
       y3_Test_PCDA=A3*y2_Test_PCDA;
       y_Test_PCDA=[y1_Test_PCDA;y3_Test_PCDA];
        
        %%%Training of CLassifier%%%
        MLE_Model=MLEBuild(y,Class_Index_Train,Class);
       
       %%%Prediction%%%
        y_pred=[];
        Prob=[];
        for i=1:size(y_Test_PCDA,2)
            Sample=y_Test_PCDA(:,i);
            [Probability,y_predicted]=Predict(Sample,MLE_Model,Class);
            y_pred=[y_pred,y_predicted];
            Prob=[Prob;Probability];
        end
        acc=0;
        for i=1:size(Class,2)
            idx=Class_Index_Test.(Class(i));
            no_correct=sum(y_pred(:,idx)==i);
            total=size(idx,2);
            acc=acc+(no_correct/total);
            
            
        end
        acc1=acc/9;
        
        %%%Average Class accuracy%%%
        accuracy=[accuracy,acc1];
        
        
      
        
    end
    
% accuracy_lambda=[accuracy_lambda;accuracy];
% end
