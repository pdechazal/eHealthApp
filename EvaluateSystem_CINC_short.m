function Performance=EvaluateSystem_CINC_short
%function Performance=EvaluateSystem_MNIST_short
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Purpose: Evaluates the Cinc apnea data
%
%
%Author: Philip de Chazal
%Date: 07-Sept-2014
%Version 1.0
%Revision History
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%     Pinv: 89.3637
%      LDA: 89.5198
     
rng(1);
FOlist=[1] %,1:10,12,15,20

%1. Load FeatureData and ExpertAnnotations
load C:/Matlab/Framework/Projects/CinC2000Apnea/Inputs/ECGapnea3avg.mat

Classlist=[78 65];
FD=[FeatureData.Data{1:2:35}]; %1:2:35
%FD(1:32,:)=exp(FD(1:32,:));
%FD(37:68,:)=exp(FD(37:68,:));
T=Class2Targets([ExpertAnnotations.Annotations{1:2:35}],Classlist);

[r,s]=find(isnan(FD));
s=unique(s);
FD(:,s)=[];
T(:,s)=[];

[d,N]=size(FD);
%scale data so zero mean and unit variance
FDmean=nanmean(FD,2);
FDstd=nanstd(FD,[],2);
FD=(FD-FDmean*ones(1,N))./(FDstd*ones(1,N));


c=size(T,1);

FD_test=[FeatureData.Data{2:2:35}]; %2:2:35
%FD_test(1:32,:)=exp(FD_test(1:32,:));
%FD_test(37:68,:)=exp(FD_test(37:68,:));
T_test=Class2Targets([ExpertAnnotations.Annotations{2:2:35}],Classlist);
[r,s]=find(isnan(FD_test));
s=unique(s);
FD_test(:,s)=[];
T_test(:,s)=[];


N_test=size(FD_test,2);
FD_test=(FD_test-FDmean*ones(1,N_test))./(FDstd*ones(1,N_test));

%clear FeatureData ExpertAnnotations




Reps=10;

ctr=0;
for FO= FOlist
    
    PP_lda=zeros(c,N_test,Reps)*nan;
    PP_pinv=zeros(c,N_test,Reps)*nan;
    ctr=ctr+1;
  
    disp(FO)
    
    %repetition
    for l=1:Reps
        clear X X_test
        disp(l)
        
        %random weights for hidden layer
        
        h=round(FO*d);
        
        W1=[(rand(h,1)-0.5)*6,(rand(h,d)-0.5)*0.1]; %uniformly distributed weights from (-1.5 to 1.5) 6,0.1
        %Implements the following X= [ones(1,N);tanh(W1*FD)];
        %X_test=[ones(1,N_test);tanh(W1*FD_test)];
        %XS=X*X'; %X Sum Squares
        % CS=X*T'; %Cross Sum Squares
        %clear X
        
        %Hidden layer outputs
      
        X=[ones(1,N);tanh(W1*[ones(1,N);FD])];%W1*FD
        X_test=[ones(1,N_test);tanh(W1*[ones(1,N_test);FD_test])];%W1*FD_test
 
        tic
        
        %Pseudoinverse
        %Linear network, SSE solution;
        %Train
        % Equivalent to W{2}=inv(X*X')(X*T')=(X*X')\(X*T');
        XS=X*X';
        CS=X*T';
        
        toc
        disp('common')
        tic
        %W2=XS\CS;
        W2=CS'/XS;
        toc
        disp('pi')
        
        W2pi=W2';
         t_pinv(l)=toc;
         %toc
       
         %Evaluate
       % Y=W2'*X_test;
  Y=W2*X_test;
 
        
%No constant model
% % %         XS=X(2:end,:)*X(2:end,:)';
% % %         CS=X(2:end,:)*T';
% % %         W2=XS\CS;
% % %         W2pi=W2';
% % %          t_pinv(l)=toc;
% % %          toc
% % %        
% % %          %Evaluate
% % %         Y=W2'*X_test(2:end,:);

        
        
        %Calculate Post probs.
        expY=exp(Y);
        
        %subtract off the mean
        Y=Y-repmat(mean(Y,1),c,1);

        PP_pinv(:,:,l)=expY./(ones(c,c)*expY);
        
        %Some output!
        temp=EvalClfrPfmEpoch(Y,T_test,Classlist,'Acc');
        disp(sprintf('   ELM+Linnet: classifier accuracy %3.3f',temp.Acc))
        Performance.Pinv=temp.Acc;
        
        
        
       % X=[tanh(W1*[zeros(1,N);FD])];%W1*FD
       % X_test=[tanh(W1*[zeros(1,N);FD_test])];%W1*FD_test
 
      
%        XS=X*X';
 %       CS=X*T';
        
        tic
        %LDA
      %  M=(X*T')/(T*T');
       % XS=X*X';
       % CS=X*T';
        TS=T*T'; %Target sum squares
        M=CS/TS; %(X*T')*inv(T*T')
       
        CV=XS-CS*M'; %CV=X(X-MT)'=XX'- (XT')M' %for single targets
       M=M(2:end,:);  %Remove first row of constants
      CV=CV(2:end,2:end)/N; %Remove first row/column of constants
        
        A=M'/CV; %A=M'*inv(CV)
        b=-.5*(A.*M')*ones(h,1)+[log(0.62); log(0.38)]; %assumed equal priors
        
        
        W2=[b,A]';
        toc
        disp('lda')
         t_lda(l)=toc;
        %        Evaluate
        Y=W2'*X_test;
        
        %subtract off the mean
        Y=Y-repmat(mean(Y,1),c,1);
        
        expY=exp(Y);
        PP_lda(:,:,l)=expY./(ones(c,c)*expY);
        
        %Some output!
        temp=EvalClfrPfmEpoch(Y,T_test,Classlist,'Acc');
        disp(sprintf('   ELM+LDA: Classifier accuracy %3.3f',temp.Acc))
        
        Performance.LDA=temp.Acc;
       
    end
    
    mPP_pinv=mean(PP_pinv,3);
    temp=EvalClfrPfmEpoch(mPP_pinv,T_test,Classlist,'Acc');
    disp(sprintf('   ELM+pinv: Combined Classifier accuracy %3.3f',temp.Acc))
    Performance.Pinv=temp.Acc;
    
    
    mPP_lda=mean(PP_lda,3);
    temp=EvalClfrPfmEpoch(mPP_lda,T_test,Classlist,'Acc');
    disp(sprintf('   ELM+LDA: Combined Classifier accuracy %3.3f',temp.Acc))
    Performance.LDA=temp.Acc;
    
    
    
    toc
    Pfm(ctr,1)=FO;
    Pfm(ctr,2)=Performance.Pinv;
    Pfm(ctr,3)=Performance.LDA;
    Pfm(ctr,4)=0;
    Pfm(ctr,5)=l;
    
   % save('c:\temp\Apnea.txt','Pfm','-ascii');
   % save(sprintf('c:\\temp\\Apnea%d',FO),'Pfm','PP_lda','mPP_lda','PP_pinv','mPP_pinv','T_test','Classlist','t_pinv','t_lda');
end



