MLE_Model=MLEBuild(y1,Class_Index_Train,Class);
y_pred=[];
Prob=[];
accuracy=[];
for i=1:size(y2,2)
    Sample=y2(:,i);
    [Probability,y_predicted]=Predict(Sample,MLE_Model,Class);
    y_pred=[y_pred,y_predicted];
    Prob=[Prob;Probability];
end



Correct_Classified=size(find(y_pred==y_test),2);
No_Test=size(y_pred,2);
accuracy=[accuracy,Correct_Classified/No_Test];




function MLE=MLEBuild(y,Class_Index_Train,Class)
for i=1:size(Class,2)
X=y(:,Class_Index_Train.(Class(i)));
Mean_X=mean(X,2);
Covariance_X=cov(X');
size(Covariance_X)
Inverse_COX=inv(Covariance_X);
MLE_Build.(Class(i)).Mean=Mean_X;
MLE_Build.(Class(i)).Covariance=Covariance_X;
MLE_Build.(Class(i)).Inverse_Cov=Inverse_COX;
end
MLE=MLE_Build;
end
%Sample=X(:,4);
%Probability=MLE(Sample,Mean_X,Inverse_COX,Covariance_X)

function [Probability,y_predicted]=Predict(Sample,MLE_Model,Class)
Probability=[];
for i=1:size(Class,2)
Mean_X=MLE_Model.(Class(i)).Mean;
Covariance_X=MLE_Model.(Class(i)).Covariance;
Inverse_COX=MLE_Model.(Class(i)).Inverse_Cov;
a=(Sample-Mean_X)'*Inverse_COX*(Sample-Mean_X)/2;
a=exp(-a);
d=size(Sample,1);
b=sqrt(((2*pi)^d)*det(Covariance_X));
Probability=[Probability,a/b];
end
[M,I]=max(Probability,[],2);
if M==0
y_predicted=0;
else
y_predicted=I;
end

end