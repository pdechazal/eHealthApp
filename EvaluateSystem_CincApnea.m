function Performance=EvaluateSystem_beat(ConfigNum)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Purpose: Evaluates a Multiple classification system for Beat project
%The system is assumed to be an epoch-based system processing different records
%
%Inputs:
%1. ConfigNum: Number of a system configuration to evaluate

%Outputs:  
%1. Performance - Performance of the system
%
%Author: Philip de Chazal
%Date: 03-Feb-2013
%Version 1.0
%Revision History
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(1);

Config=LoadConfig_CincApnea(ConfigNum);
[FeatureData,ExpertAnnotations]=GenerateData_beat(Config);



if ~isempty(Config.ClassCombine)
   
    for i=1:length(Config.ClassCombine)
      for j=1:length(Config.ClassCombine{i})
         for k=1:length(ExpertAnnotations.Annotations)
            indx=find(ExpertAnnotations.Annotations{k}==Config.ClassCombine{i}(j));
            ExpertAnnotations1{k}(indx)=Config.ClassList(i);
         end
      end
   end
   ExpertAnnotations.Annotations=ExpertAnnotations1;
   ExpertAnnotations.Codes=Config.ClassList;
   ExpertAnnotations.Description=Config.ClassCombine;
   clear ExpertAnnotations1;
end

%Replace missing values with class means 

if strcmp(Config.MissingValueProcessing,'ClassMean') 

%Do nothing as no mising values for CInc200 apnea set
    
end


%Shift Features if required
L=size(FeatureData.Data{1},1)
for i=1:length(Config.FeatureShift)
    Shift=Config.FeatureShift(i);
   %Shift Features
   for j=1:length(FeatureData.Data)
      if Shift>0 
         %%%%%%%%%%
         %Use features in the past to predict the present 
         %%%%%%%%%%%%%%
         FeatureData.Data{j}([1:L]+i*L,:)=FeatureData.Data{j}(1:L,[ones(1,Shift),1:end-Shift]); %+Shift
     else
         %%%%%%%%%%
         %Use features from the future to predict the present 
         %%%%%%%%%%%%%%
         E=size(FeatureData.Data{j},2);
         FeatureData.Data{j}([1:L]+i*L,:)=FeatureData.Data{j}(1:L,[(abs(Shift)+1):E,ones(1,abs(Shift))*E]); %-Shift
      end
   
     %Examples
   %   FeatureData.Data{i}([1:L]+i*L,:)=FeatureData.Data{i}(1:L,[1,1:end-1]); %+1 shift
   %   FeatureData.Data{i}([1:L]+i*L,:)=FeatureData.Data{i}(1:L,[2:end,end]); %-1 shift
   %   FeatureData.Data{i}([1:L]+i*L,:)=FeatureData.Data{i}(1:L,[1,1,1:end-2]); %+2 shift
   %   FeatureData.Data{i}([1:L]+i*L,:)=FeatureData.Data{i}(1:L,[3:end,end,end]); %-2 shift
   %   FeatureData.Data{i}([1:L]+i*L,:)=FeatureData.Data{i}(1:L,[1,1,1,1:end-3]); %+3 shift
   %   FeatureData.Data{i}([1:L]+i*L,:)=FeatureData.Data{i}(1:L,[4:end,end,end,end]); %-3 shift
   end
end


DataSplits=GenerateDataSplits_CincApnea(Config,size(FeatureData,1));
Performance=EvaluateDataSplits_CincApnea(Config,FeatureData,ExpertAnnotations,DataSplits);

   
save(Config.OutputPerformanceFile,'Performance','Config');

%Data Structures
%FeatureData{1:NumRecord,1:NumChannel}(1:NumFeatures,1:NumEpochs)
%ExpertAnnotations{1:NumRecords}
%DataSplits(1:NumDataSplits).Train(1:NumTrainingExamples)
%                           .Test(1:NumTestingExamples
%Performance.Train
%           .Test


                            