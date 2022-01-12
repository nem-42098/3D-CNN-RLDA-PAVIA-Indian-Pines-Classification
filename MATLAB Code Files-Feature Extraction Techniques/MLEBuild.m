function [MLE]=MLEBuild(y,Class_Index_Train,Class)
for i=1:size(Class,2)
X=y(:,Class_Index_Train.(Class(i)));
Mean_X=mean(X,2);
Covariance_X=cov(X');
Inverse_COX=inv(Covariance_X);
MLE_Build.(Class(i)).Mean=Mean_X;
MLE_Build.(Class(i)).Covariance=Covariance_X;
MLE_Build.(Class(i)).Inverse_Cov=Inverse_COX;
end
MLE=MLE_Build;
end