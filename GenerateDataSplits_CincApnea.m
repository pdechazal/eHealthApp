function DataSplits=GenerateDataSplits(Config,NumFolds);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Purpose: Generates training and testing data
%
%Inputs:
%1. Config
%2. NumFolds: Number of Folds

%Outputs:  
%1. DataSplits(NumFolds).Train
%                       .Test
%
%Author: Philip de Chazal
%Date: 14-Oct-2005
%Version 1.0
%Revision History
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch Config.DataSplits
case 'ReleasedSetXV'
   
   NumFolds=35;
   for i=1:NumFolds
      DataSplits(i).Train=[1:i-1,i+1:NumFolds];
      DataSplits(i).Test=i;
   end

case 'ReleasedSetResub'
   
   DataSplits.Train=[1:35];
   DataSplits.Test=[1:35];

   
case 'ReleasedSetResub2'
%    DataSplits.Train=[1,3,4,5,7,8,9,10,11,...
%        13,14,15,16,18,19,20,21,23,24,...
%        26,27,28,29,30,31,32,33,34,35];
%    DataSplits.Test=DataSplits.Train;

    DataSplits.Train=[1,14,19,23];
    DataSplits.Test=DataSplits.Train;

case 'ReleasedSetXV2'
   
   Set=[1,3,4,5,7,8,9,10,11,...
       13,14,15,16,18,19,20,21,23,24,...
       26,27,28,29,30,31,32,33,34,35];
   
   NumFolds=29;
   for i=1:NumFolds
      DataSplits(i).Train=Set([1:i-1,i+1:NumFolds]);
      DataSplits(i).Test=Set(i);
   end

case 'WithheldSet'
       DataSplits.Train=[1:35];
   DataSplits.Test=[36:70];

otherwise
   error('Unrecognised datasplit')
end