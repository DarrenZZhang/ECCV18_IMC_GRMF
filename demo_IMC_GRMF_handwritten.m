% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me via: wenjie@hrbeu.edu.cn 
% If you find the code is useful, please cite the following reference:
% Jie Wen , Zheng Zhang, Yong Xu, Zuofeng Zhong, Bob Zhang, Lunke Fei, 
% Incomplete Multi-view Clustering via Graph Regularized Matrix Factorization [C], 
% European Conference on Computer Vision Workshop on Compact and Efficient Feature Representation and Learning in Computer Vision, 2018.
% homepage: https://sites.google.com/view/jerry-wen-hit/publications
% Note: please set the suitable lambda1, lambda2 and options.k to implement
% the algorithm. Meanwile, the result is sensitive to the kmeans algorithm,
% so the results may have some deviations with the paper.
clear;
clc
Dataname = 'handwrittenRnsp';
Datafold = 'handwrittenFolds';
num_view = 2;
load(Dataname);
load(Datafold);
[numFold,numInst]=size(folds);

% % pairPortion = 0.1;
% % lambda1 = 1e2;
% % lambda2 = 1e-1;
% % kkk = 10;                   % parameter nearest neighbor size

pairPortion = 0.3;
lambda1 = 1e2;
lambda2 = 1e-3;
kkk = 10;

% % pairPortion = 0.5;
% % lambda1 = 1e2;
% % lambda2 = 1e-3;
% % kkk = 10;

% % pairPortion = 0.7;
% % lambda1 = 1e2;
% % lambda2 = 1e-3;
% % kkk = 10;

% % pairPortion = 0.9;
% % lambda1 = 1e2;
% % lambda2 = 1e-3;
% % kkk = 10;

f = 4;
instanceIdx = folds(f,:);
truthF      = truth(instanceIdx);
X1 = NormalizeFea(X1,1);
X2 = NormalizeFea(X2,1);
X{1} = X1;  
X{2} = X2; 
clear X1 X2 truth folds  numFold     
numpairedInst = floor(numInst*pairPortion);  % number of paired instances have all views
paired        = instanceIdx(1:numpairedInst); 
singledNumView1 = ceil(0.5*(length(instanceIdx)-numpairedInst)); % half of remaining samples as the first view
singleInstView1 = instanceIdx(numpairedInst+1:numpairedInst+singledNumView1);   % index of the first single view
singleInstView2 = instanceIdx(numpairedInst+singledNumView1+1:end);             % index of the second single view
singledNumView2 = length(singleInstView2);              % number of the second single view
xpaired = X{1}(paired,:);
ypaired = X{2}(paired,:);
xsingle = X{1}(singleInstView1,:);
ysingle = X{2}(singleInstView2,:);
clear singleInstView1 singleInstView2 paired

X1 = [xpaired;xsingle];    
X2 = [ypaired;ysingle];
clear xpaired xsingle ypaired ysingle

options = [];
options.NeighborMode = 'KNN';
options.k = kkk;                     % parameter of the nearest neighbor number 
options.WeightMode = 'Binary';      % Binary  HeatKernel
Z1 = constructW(X1,options);
W{1} = full(Z1);
clear Z1;
Z2 = constructW(X2,options);
W{2} = full(Z2);
clear Z2;                    
X{1} = X1;
X{2} = X2;
clear X1 X2

G1 = diag([ones(1,numpairedInst),zeros(1,singledNumView1)]);
G1(numpairedInst+1:end,:) = [];
G2 = diag([ones(1,numpairedInst),zeros(1,singledNumView2)]);
G2(numpairedInst+1:end,:) = [];  
G{1} = G1;
G{2} = G2;
clear G1 G2

opts.lambda1 = lambda1;
opts.lambda2 = lambda2;
opts.nnClass = numClust;
opts.num_view= num_view;
opts.max_iter= 100;
[Pc,P,Q,obj] = IMC_GRMF(X,W,G,opts);
new_F    = [Pc;P{1}(numpairedInst+1:end,:);P{2}(numpairedInst+1:end,:)];
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
% avoid divide by zero
for i = 1:size(norm_mat,1)
    if (norm_mat(i,1)==0)
        norm_mat(i,:) = 1;
    end
end
new_F = new_F./norm_mat; 

for iter_c = 1:5
    pre_labels    = kmeans(new_F,numClust,'emptyaction','singleton','replicates',20,'display','off');
    result_LatLRR = ClusteringMeasure(truthF, pre_labels);       
    AC(iter_c)    = result_LatLRR(1)*100;
    MIhat(iter_c) = result_LatLRR(2)*100;
    Purity(iter_c)= result_LatLRR(3)*100;
end
mean_ACC = mean(AC)
mean_NMI = mean(MIhat)
mean_PUR = mean(Purity)







