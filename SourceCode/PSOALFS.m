function [Output]=PSOALFS(Input)
Input1.datain=Input.data0;
Input1.dataout=Input.Y0;
[Output]=ALFS(Input1,'Learning');
[Output.C,Output.A,Output.LX,Output.S]=PSOCA(Output.C,Output.A,Input.data0,Input.Y0,Output.MN,Input.MaxIt,Input.nPop,Input.w,Input.cs,Input.cp,Input.wdamp);
Input2=Output;
Input2.datain=Input.data1;
Input2.dataout=Input.Y1;
[Output2]=ALFS(Input2,'Testing');
Output.RMSE=sqrt(mean((Output2.Ye-Input.Y1).^2));
Output.PreSeq=Output2.Ye;
end
function [Output]=ALFS(Input,Mode)
if strcmp(Mode,'Learning')==1
    [Output.MN,Output.A,Output.C,Output.LX,Output.S,Output.Cov,Output.Index,Output.sum_lambda,Output.Global_mean,Output.Global_X,Output.Ye,Output.L]=LearningALM(Input.datain,Input.dataout);
end
if strcmp(Mode,'Updating')==1
    [Output.MN,Output.A,Output.C,Output.LX,Output.S,Output.Cov,Output.Index,Output.sum_lambda,Output.Global_mean,Output.Global_X,Output.Ye,Output.L]=UpdatingALM(Input.datain,Input.dataout,Input.MN,Input.A,Input.C,Input.LX,Input.S,Input.Cov,Input.Index,Input.sum_lambda,Input.Global_mean,Input.Global_X,Input.L);
end
if strcmp(Mode,'Testing')==1
    [Output.Ye]=TestingALM(Input.datain,Input.MN,Input.A,Input.C,Input.LX,Input.S);
end
end
function [ModelNumber,A,center,Local_X,Support,C,Index,sum_lambda,Global_mean,Global_X,Ye,L]=LearningALM(data0,Y0)
threshold=exp(-1/8);
omega=10;
forgettingfactor=0.1;
[L,W]=size(data0);
center=data0(1,:);
Global_mean=data0(1,:);
Global_X=sum(data0(1,:).^2);
Local_X=sum(data0(1,:).^2);
Support=1;
ModelNumber=1;
sum_lambda=1;
Index=1;
A=zeros(1,W+1);
C=eye(W+1)*omega;
Ye=zeros(L,1);
%%
for ii=2:1:L
    datain=data0(ii,:);
    yin=Y0(ii,:);
    Global_mean=Global_mean.*(ii-1)/ii+datain./ii;
    Global_X=Global_X.*(ii-1)/ii+sum(datain.^2)/ii;
    datadensity=1/(sum((datain-Global_mean).^2)+Global_X-sum(Global_mean.^2)+0.000000001);
    centerdensity=zeros(ModelNumber,1);
    for jj=1:1:ModelNumber
        centerdensity(jj)=1/(sum((center(jj,:)-Global_mean).^2)+Global_X-sum(Global_mean.^2)+0.000000001);
    end
    [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support);
    Ye(ii)=[1,datain]*A'*centerlambda;
    if datadensity>max(centerdensity)||datadensity<min(centerdensity)
        seq=find(LocalDensity>threshold);
        if isempty(seq)~=1
            OL=1;
            [~,b]=max(LocalDensity(seq));
            target=seq(b);
        else
            OL=0;
            target=0;
        end
        %%
        if OL==1
            seq=1:1:ModelNumber;
            seq(target)=[];
            Index=Index(seq);
            Index(end+1)=ii;
            sum_lambda=sum_lambda(seq);
            sum_lambda(end+1)=0;
            centerlambda=centerlambda(seq)./sum(centerlambda(seq));
            centerlambda(end+1)=0;
            A0=A(target,:);
            A=A(seq,:);
            A(end+1,:)=A0;
            C=C(:,:,seq);
            C(:,:,end+1)=eye(W+1)*omega;
            center0=center(target,:);
            center=center(seq,:);
            center(end+1,:)=(datain+center0)/2;
            Local_X0=Local_X(target);
            Local_X=Local_X(seq);
            Local_X(end+1)=(sum(datain.^2)+Local_X0)/2;
            Support0=Support(target);
            Support=Support(seq);
            Support(end+1)=ceil((1+Support0)/2);
        else
            %% new cloud add
            ModelNumber=ModelNumber+1;
            center=[center;datain];
            Support=[Support,1];
            Local_X=[Local_X,sum(datain.^2)];
            sum_lambda=[sum_lambda,0];
            Index=[Index,ii];
            A(ModelNumber,:)=mean(A,1);
            C(:,:,ModelNumber)=eye(W+1)*omega;
        end
    else
        %% local_parameters_update
        [~,label0]=min(pdist2(datain,center));
        Support(label0)=Support(label0)+1;
        center(label0,:)=(Support(label0)-1)/Support(label0)*center(label0,:)+datain/Support(label0);
        Local_X(label0)=(Support(label0)-1)/Support(label0)*Local_X(label0)+sum(datain.^2)/Support(label0);
    end
    %%
    [centerlambda,~]=firingstrength(datain,ModelNumber,center,Local_X,Support);
    %% stale_datacloud_remove
    sum_lambda=sum_lambda+centerlambda';
    utility=zeros(ModelNumber,1);
    for jj=1:1:ModelNumber
        if Index(jj)~=ii
            utility(jj)=sum_lambda(jj)/(ii-Index(jj));
        else
            utility(jj)=1;
        end
    end
    seq=find(utility>=forgettingfactor);
    ModelNumber0=length(seq);
    if ModelNumber0<ModelNumber
        center=center(seq,:);
        Local_X=Local_X(seq);
        Index=Index(seq);
        sum_lambda=sum_lambda(seq);
        centerlambda=centerlambda(seq)./sum(centerlambda(seq));
        Support=Support(seq);
        A=A(seq,:);
        C=C(:,:,seq);
    end
    ModelNumber=ModelNumber0;
    X=[1,datain];
    for jj=1:1:ModelNumber
        C(:,:,jj)=C(:,:,jj)-centerlambda(jj)*C(:,:,jj)*X'*X*C(:,:,jj)/(1+centerlambda(jj)*X*C(:,:,jj)*X');
        A1=A(jj,:)'+centerlambda(jj)*C(:,:,jj)*X'*(yin-X*A(jj,:)');
        A(jj,:)=A1';
    end
end
end
function [ModelNumber,A,center,Local_X,Support,C,Index,sum_lambda,Global_mean,Global_X,Ye,L0]=UpdatingALM(data0,Y0,ModelNumber,A,center,Local_X,Support,C,Index,sum_lambda,Global_mean,Global_X,L0)
threshold=exp(-1/8);
omega=10;
forgettingfactor=0.1;
[L,W]=size(data0);
Ye=zeros(L,1);
%%
for ii=L0+1:1:L+L0
    datain=data0(ii-L0,:);
    yin=Y0(ii-L0,:);
    Global_mean=Global_mean.*(ii-1)/ii+datain./ii;
    Global_X=Global_X.*(ii-1)/ii+sum(datain.^2)/ii;
    datadensity=1/(sum((datain-Global_mean).^2)+Global_X-sum(Global_mean.^2)+0.000000001);
    centerdensity=zeros(ModelNumber,1);
    for jj=1:1:ModelNumber
        centerdensity(jj)=1/(sum((center(jj,:)-Global_mean).^2)+Global_X-sum(Global_mean.^2)+0.000000001);
    end
    [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support);
    Ye(ii-L0)=[1,datain]*A'*centerlambda;
    if datadensity>max(centerdensity)||datadensity<min(centerdensity)
        seq=find(LocalDensity>threshold);
        if isempty(seq)~=1
            OL=1;
            [~,b]=max(LocalDensity(seq));
            target=seq(b);
        else
            OL=0;
            target=0;
        end
        %%
        if OL==1
            seq=1:1:ModelNumber;
            seq(target)=[];
            Index=Index(seq);
            Index(end+1)=ii;
            sum_lambda=sum_lambda(seq);
            sum_lambda(end+1)=0;
            centerlambda=centerlambda(seq)./sum(centerlambda(seq));
            centerlambda(end+1)=0;
            A0=A(target,:);
            A=A(seq,:);
            A(end+1,:)=A0;
            C=C(:,:,seq);
            C(:,:,end+1)=eye(W+1)*omega;
            center0=center(target,:);
            center=center(seq,:);
            center(end+1,:)=(datain+center0)/2;
            Local_X0=Local_X(target);
            Local_X=Local_X(seq);
            Local_X(end+1)=(sum(datain.^2)+Local_X0)/2;
            Support0=Support(target);
            Support=Support(seq);
            Support(end+1)=ceil((1+Support0)/2);
        else
            %% new cloud add
            ModelNumber=ModelNumber+1;
            center=[center;datain];
            Support=[Support,1];
            Local_X=[Local_X,sum(datain.^2)];
            sum_lambda=[sum_lambda,0];
            Index=[Index,ii];
            A(ModelNumber,:)=mean(A,1);
            C(:,:,ModelNumber)=eye(W+1)*omega;
        end
    else
        %% local_parameters_update
        [~,label0]=min(pdist2(datain,center));
        Support(label0)=Support(label0)+1;
        center(label0,:)=(Support(label0)-1)/Support(label0)*center(label0,:)+datain/Support(label0);
        Local_X(label0)=(Support(label0)-1)/Support(label0)*Local_X(label0)+sum(datain.^2)/Support(label0);
    end
    %%
    [centerlambda,~]=firingstrength(datain,ModelNumber,center,Local_X,Support);
    %% stale_datacloud_remove
    sum_lambda=sum_lambda+centerlambda';
    utility=zeros(ModelNumber,1);
    for jj=1:1:ModelNumber
        if Index(jj)~=ii
            utility(jj)=sum_lambda(jj)/(ii-Index(jj));
        else
            utility(jj)=1;
        end
    end
    seq=find(utility>=forgettingfactor);
    ModelNumber0=length(seq);
    if ModelNumber0<ModelNumber
        center=center(seq,:);
        Local_X=Local_X(seq);
        Index=Index(seq);
        sum_lambda=sum_lambda(seq);
        centerlambda=centerlambda(seq)./sum(centerlambda(seq));
        Support=Support(seq);
        A=A(seq,:);
        C=C(:,:,seq);
    end
    ModelNumber=ModelNumber0;
    X=[1,datain];
    for jj=1:1:ModelNumber
        C(:,:,jj)=C(:,:,jj)-centerlambda(jj)*C(:,:,jj)*X'*X*C(:,:,jj)/(1+centerlambda(jj)*X*C(:,:,jj)*X');
        A1=A(jj,:)'+centerlambda(jj)*C(:,:,jj)*X'*(yin-X*A(jj,:)');
        A(jj,:)=A1';
    end
end
L0=L+L0;
end
function [Ye]=TestingALM(data1,ModelNumber,A,center,Local_X,Support)
[L,W]=size(data1);
Ye=zeros(L,1);
seq=find(Support~=0);
ModelNumber=length(seq);
for ii=1:1:L
    datain=data1(ii,:);
    [centerlambda,~]=firingstrength(datain,ModelNumber,center(seq,:),Local_X(seq),Support(seq));
    Ye(ii)=[1,datain]*A(seq,:)'*centerlambda;
end
end
function Ye=systemoutput(datain,A,centerlambda)
Ye=[1,datain]*A'*centerlambda;
end
function [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support)
a0=Support./(Support+1);
a1=1./(Support+1);
center1=center;
Local_X1=Local_X;
LocalDensity=zeros(ModelNumber,1);
for jj=1:1:ModelNumber
    center1(jj,:)=center(jj,:)*a0(jj)+datain*a1(jj);
    Local_X1(jj)=Local_X(jj)*a0(jj)+sum(datain.^2)*a1(jj);
    AA=(Local_X1(jj)-sum(center1(jj,:).^2));
    BB=sum((datain-center1(jj,:)).^2);
    if AA==0
        LocalDensity(jj)=1;
    else
        LocalDensity(jj)=BB/AA;
    end
end
LocalDensity=exp(-0.5*LocalDensity);
centerlambda=LocalDensity./sum(LocalDensity);
end
function [output1,output2,output3,output4]=PSOCA(center,A,data,Y,ModelNumber,MaxIt,nPop,w,cs,cp,wdamp)
Dma=[max(abs(center),[],1),max(abs(A),[],1)];
Obj=[center,A];
Particles=[];
Particles.position=[];
Particles.velocity=[];
Particles=repmat(Particles,nPop,1);
cost=zeros(nPop,1);
Parbest.position=[];
Parbestcost=zeros(nPop,1);
Parbest=repmat(Parbest,nPop,1);
[L,W]=size(Obj);
WC=(W-1)/2;
GlobalBest.Cost=[];
GlobalBest.Position=[];
GlobalBest.LX=[];
GlobalBest.S=[];
vmax=(Dma)*0.2;
vmin=vmax*-1;
Local_X=zeros(nPop,ModelNumber);
Support=zeros(nPop,ModelNumber);
%%
ii=1;
Particles(ii).position=Obj;
Particles(ii).velocity=zeros(L,W);
[cost(ii),Local_X(ii,:),Support(ii,:),Particles(ii).position(:,1:1:WC)]=CostFunct(data,Y,ModelNumber,Particles(ii).position(:,WC+1:1:end),Particles(ii).position(:,1:1:WC));
Parbest(ii).position=Particles(ii).position;
Parbestcost(ii)=cost(ii);
for ii=2:1:nPop
    Particles(ii).position=Obj+(2*rand(L,W)-1).*repmat(vmax,L,1);
    Particles(ii).velocity=zeros(L,W);
    [cost(ii),Local_X(ii,:),Support(ii,:),Particles(ii).position(:,1:1:WC)]=CostFunct(data,Y,ModelNumber,Particles(ii).position(:,WC+1:1:end),Particles(ii).position(:,1:1:WC));
    Parbest(ii).position=Particles(ii).position;
    Parbestcost(ii)=cost(ii);
end
GBC=[];
[GlobalBest.Cost,jj]=min(cost);
GBC=GlobalBest.Cost;
GlobalBest.Position=Particles(jj).position;
GlobalBest.LX=Local_X(jj,:);
GlobalBest.S=Support(jj,:);
for it=1:1:MaxIt
    for ii=1:1:nPop
        Particles(ii).velocity=Particles(ii).velocity*w+cp*rand(1,W).*(Parbest(ii).position-Particles(ii).position)+cs*rand(1,W).*(GlobalBest.Position-Particles(ii).position);
        Particles(ii).velocity=max(Particles(ii).velocity,vmin);
        Particles(ii).velocity=min(Particles(ii).velocity,vmax);
        Particles(ii).position=Particles(ii).position+Particles(ii).velocity;
        [cost(ii),Local_X(ii,:),Support(ii,:),Particles(ii).position(:,1:1:WC)]=CostFunct(data,Y,ModelNumber,Particles(ii).position(:,WC+1:1:end),Particles(ii).position(:,1:1:WC));
        if cost(ii)<Parbestcost(ii)
            Parbest(ii).position=Particles(ii).position;
            Parbestcost(ii)=cost(ii);
            if Parbestcost(ii)<GlobalBest.Cost
                GlobalBest.Position=Parbest(ii).position;
                GlobalBest.Cost=Parbestcost(ii);
                GlobalBest.LX=Local_X(ii,:);
                GlobalBest.S=Support(ii,:);
            end
        end
    end
    GBC=[GBC;GlobalBest.Cost];
    w=wdamp*w;
end
output1=GlobalBest.Position(:,1:1:WC);
output2=GlobalBest.Position(:,WC+1:1:end);
output3=GlobalBest.LX;
output4=GlobalBest.S;
end
function [F1,Local_X,Support,center]=CostFunct(data1,Y1,ModelNumber,A,center)
[~,x]=min(pdist2(center,data1),[],1);
Support=zeros(1,ModelNumber);
Local_X=zeros(1,ModelNumber);
for ii=1:1:ModelNumber
    seq=find(x==ii);
    Local_X(ii)=mean(sum(data1(seq,:).^2,2));
    Support(ii)=length(seq);
    if Support(ii)~=0
        center(ii,:)=mean(data1(seq,:),1);
    else
        center(ii,:)=0;
    end
end
[L,W]=size(data1);
Ye=zeros(L,1);
for ii=1:1:L
    datain=data1(ii,:);
    [centerlambda,~]=firingstrength(datain,ModelNumber,center,Local_X,Support);
    Ye(ii)=systemoutput(datain,A,centerlambda);
end
F1=sqrt(mean((Ye-Y1).^2));
end
