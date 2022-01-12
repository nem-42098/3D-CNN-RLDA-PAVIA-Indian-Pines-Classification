function [Class_Index] =Compute_Classindex(GroundTruth,Class)
    No_class=max(GroundTruth,[],'all');
       for i=1:No_class
        [row,col]=find( GroundTruth == i);
        Index=[];
        Index=[Index ;  col];
        Data.(Class(i))=Index;
       end
    Class_Index=Data;
end