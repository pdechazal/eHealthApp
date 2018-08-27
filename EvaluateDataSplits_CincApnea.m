function Performance=EvaluateDataSplits(Config,FeatureData,ExpertAnnotations,DataSplits)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Purpose: Trains and tests a system using provided datasplits (train/test data)
%
%Sizes
%  d: Number of Features 
%  N: Number of Cases in a record
%  R: Number of Recordings
%  C: Number of Channels
%  c: Number of Classes
%
%Inputs:
%1. Config 
%2. FeatureData.Data{R}[d,N]
%                 .Names{d}
%3. ExpertAnnotations.Annotation{R}[N]
%                       .Codes(c)
%                       .Description{c}
%4. DataSplits(NumFolds).Train[1,N1] %N1,N2,N1+N2<=N
%                       .Test[1,N2]

%Outputs:  
%1. Performance 
%

%Other Structrs used
%1. ExpertTargets.Targets{R}[c,N]
%                   .ClassOrder[c]
%2. ChannelPP{R}[c,N]
%3. PredictedPP{R}[c,N]
%
%Author: Philip de Chazal
%Date: 18-Oct-2005
%Version 1.0
%Revision History
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumDataSplits=length(DataSplits);
%Generate target vectors from the expert annotations
for i=1:Config.NumRecords
   [ExpertTargets.Targets{i},ExpertTargets.ClassOrder]=Class2Targets(ExpertAnnotations.Annotations{i},Config.ClassList);
end

Priors=Config.Priors';

for i=1:NumDataSplits
   disp(sprintf('Training/Testing using data split %d of %d',i,NumDataSplits))
   
   TestingRecords=DataSplits(i).Test;
   
   for j=1:Config.NumStreams
      Classifier{j}=TrainClassifier(FeatureData,ExpertTargets,DataSplits(i),Config.Features{j},Config.ClassifierType{j},Config.Clfr);
      ChannelPP{i}(:,:,j)=CalculatePP([FeatureData.Data{TestingRecords}],Classifier{j},Priors);
%         st.PostProbAvgLength=3;
%   ChannelPP{i}(1,:,j)=AvgPostProb(ChannelPP{i}(1,:,j),st);
%   ChannelPP{i}(2,:,j)=AvgPostProb(ChannelPP{i}(2,:,j),st);
%      
      %Some output!
      temp=EvalClfrPfmEpoch(ChannelPP{i}(:,:,j),[ExpertTargets.Targets{TestingRecords}],ExpertTargets.ClassOrder,'Acc');
      disp(sprintf('   Channel %d: Classifier accuracy %3.1f',j,temp.Acc))
      
   end
   PredictedPP{i}=LateIntegrationPP(ChannelPP{i},Config.LateIntegrationMethod,Config.LIM);
   

  Invalid=find(all(isnan(PredictedPP{i}),1));
   PredictedPP{i}(1,Invalid)=Config.Priors(1);
   PredictedPP{i}(2,Invalid)=Config.Priors(2);
 
   PfmRecord(i)=EvalClfrPfmEpoch(PredictedPP{i},[ExpertTargets.Targets{TestingRecords}],ExpertTargets.ClassOrder,Config.PerformanceEvaluationMethod,Config.PEM);
   
   %Some output!
   temp=EvalClfrPfmEpoch(PredictedPP{i},[ExpertTargets.Targets{TestingRecords}],ExpertTargets.ClassOrder,'Acc');
   disp(sprintf('   Combine:. Classifier accuracy %3.1f',temp.Acc))
   
   
end

%Final output
Performance=EvalClfrPfmEpoch([PredictedPP{:}],[ExpertTargets.Targets{[DataSplits(:).Test]}],ExpertTargets.ClassOrder,Config.PerformanceEvaluationMethod,Config.PEM);

AllPP=[PredictedPP{:}];
AllT=[ExpertTargets.Targets{[DataSplits(:).Test]}];
%Thresholding
ctr=0;
for PPthr=0.5:0.01:1
    ctr=ctr+1;
    idx=find(AllPP(1,:)>PPthr | AllPP(1,:)<1-PPthr);
    PerformanceTemp=EvalClfrPfmEpoch(AllPP(:,idx),AllT(:,idx),ExpertTargets.ClassOrder,Config.PerformanceEvaluationMethod,Config.PEM);
    AccThr(ctr)=PerformanceTemp.Acc;
end

%Channel outputs
AllChanPP=[ChannelPP{:}];
for j=1:Config.NumStreams
    Performance.Channel(j)=EvalClfrPfmEpoch(AllChanPP(:,:,j),[ExpertTargets.Targets{[DataSplits(:).Test]}],ExpertTargets.ClassOrder,Config.PerformanceEvaluationMethod,Config.PEM);
    disp(sprintf('Channel %d: overall classifier accuracy = %3.1f',j,Performance.Channel(j).Acc))
end   
disp(sprintf('Overall classifier accuracy = %3.1f',Performance.Acc))
Performance.Record=PfmRecord;

%
