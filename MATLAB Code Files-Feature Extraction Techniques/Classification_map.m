%%%Classification map%%%
a=size(Dataset_mat);
rgb_dataset=zeros(a(1),a(2),3);
c=size(GroundTruth,2);
Pred_Ground=zeros(1,c);

y1=A1*Dataset;
y2=A2*Dataset;
y3=A3*y2;
y=[y1;y3];

for i=1:size(Class,2)
    index=Class_Index_Complete.(Class(i));
    for j=1:size(index,2)
        [Probability,y_predicted]=Predict(y(:,index(j)),MLE_Model,Class);
        Pred_Ground(1,index(j))=y_predicted;
    end
end
Pred_Ground=reshape(Pred_Ground,a(1),a(2));

color=[[1,1,1];[1,0,0];[0,1,0];[0,0,1];[1,1,0];[1,0,1];[0,1,1];[0, 0.4470, 0.7410];[0.4660, 0.6740, 0.1880];[0.6350, 0.0780, 0.1840];[0.2,0.5,0.5]];
for i=1:a(1)
    for j=1:a(2)
        z=double(Pred_Ground(i,j));
        if z==0
            rgb_dataset(i,j,:)=color(z+1,:);
        else
            rgb_dataset(i,j,:)=color(z+1,:);
        end
    end
end
imshow(rgb_dataset(:,:,:))

text(350,1,Class(1),'Color',color(2,:),'FontSize',10)
text(350,30,Class(2),'Color',color(3,:),'FontSize',10)
text(350,60,Class(3),'Color',color(4,:),'FontSize',10)
text(350,90,Class(4),'Color',color(5,:),'FontSize',10)
text(350,120,Class(5),'Color',color(6,:),'FontSize',10)
text(350,150,Class(6),'Color',color(7,:),'FontSize',10)
text(350,180,Class(7),'Color',color(8,:),'FontSize',10)
text(350,210,Class(8),'Color',color(9,:),'FontSize',10)
text(350,240,Class(9),'Color',color(10,:),'FontSize',10)
%text(350,270,Class(10),'Color',color(11,:),'FontSize',10)

