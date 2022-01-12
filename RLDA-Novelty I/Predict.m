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