function [Pc,P,U,obj] = IMC_GRMF(X,W,G,opts)
% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me via: wenjie@hrbeu.edu.cn 
% If you find the code is useful, please cite the following reference:
% Jie Wen , Zheng Zhang, Yong Xu, Zuofeng Zhong, Bob Zhang, Lunke Fei, 
% Incomplete Multi-view Clustering via Graph Regularized Matrix Factorization [C], 
% European Conference on Computer Vision Workshop on Compact and Efficient Feature Representation and Learning in Computer Vision, 2018.
% homepage: https://sites.google.com/view/jerry-wen-hit/publications
num_view   = opts.num_view;
num_sample = size(X{1},2);
lambda1    = opts.lambda1;
lambda2    = opts.lambda2;
nnClass    = opts.nnClass;
max_iter   = opts.max_iter;
Pc = 0;
for k = 1:num_view
    D{k} = diag(sum(W{k}));
    U{k} = rand(nnClass,size(X{k},2));
    P{k} = rand(size(X{k},1),nnClass);
    Pc   = Pc + G{k}*P{k};
end
Pc = Pc/num_view;
for iter = 1:max_iter
    linshi_Pc = 0;
    for k = 1:num_view
        % --------------- Uk --------------- %
        [Gs,~,Vs] = svd(X{k}'*W{k}*P{k},'econ');
        Gs(isnan(Gs)) = 0;
        Vs(isnan(Vs)) = 0;
        U{k} = Vs*Gs';  
        clear Vs Gs
        % -------------- Pk -------------- % 
        M = D{k}+lambda1*G{k}'*G{k};
        A = U{k}*X{k}'*W{k}+lambda1*Pc'*G{k};
        C = (A*diag(1./(diag(M))))';
        linshi_P = [];
        for ip = 1:size(P{k},1)            
            temp1 = 0.5*lambda2/M(ip,ip);
            temp2 = C(ip,:);
            linshi_P(ip,:) = max(0,temp2-temp1) + min(0,temp2+temp1);
        end
        P{k} = linshi_P;              
        linshi_Pc = linshi_Pc + G{k}*P{k};     
    end
    % -----------------  Pc --------------- %
    Pc = linshi_Pc/num_view;
    % ------------- obj ------------- %
    linshi_obj = 0;
    for k = 1:num_view
        linshi_obj = linshi_obj + trace(X{k}'*D{k}*X{k})+trace(P{k}'*D{k}*P{k})-2*trace(X{k}'*W{k}*P{k}*U{k})+lambda1*norm(G{k}*P{k}-Pc,'fro')^2+lambda2*sum(abs(P{k}(:)));
    end
    obj(iter) = linshi_obj;
    if iter > 2 && abs(obj(iter)-obj(iter-1))<1e-7
        iter
        break;
    end
end
end