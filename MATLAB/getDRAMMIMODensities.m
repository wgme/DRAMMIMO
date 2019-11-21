%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By: Wei Gao (wg14@my.fsu.edu)
% Last Modified: 11/20/2019
% Desciption:
% 1. Based on the code from Dr. Marko Laine 
%    (http://helios.fmi.fi/~lainema/mcmc/).
% 2. Also based on the math from Dr. Ralph C. Smith 
%    (Uncertainty Quantification: Theory, Implementation, and Applications).
% V01: Null.
% V02a: Null.
% V02b: Null.
% V03: The code was a simplified version of the density() function in 
%      Dr. Marko Laine's MCMC toolbox for MATLAB. Refer to it for details.
% V04: The code for saving results was added.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% qChain: matrix, Ms * p.
%         Ms = number of iterations selected.
%         p = number of parameters estimated.
% qVals: matrix, 100 * p. Parameter values that have posterior densities.
% qProbs: matrix, 100 * p. Corresponding posterior densities.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [qVals,qProbs] = getDRAMMIMODensities(qChain)

    disp('--------------------------------------------------');
    disp('Initializing posterior densities...');
    
    % Initialization
    numIterations = size(qChain,1);
    numChains = size(qChain,2);
    numNodes = 100;
    qMin = min(qChain,[],1);
    qMax = max(qChain,[],1);
    qRange = qMax-qMin;
    qVals = zeros(numNodes,numChains);
    qProbs = zeros(size(qVals));
    
    disp('Generating posterior densities...');
    
    % Kernel density estimation (Gaussian).
    for i = 1:1:numChains
        qVals(:,i) = linspace(qMin(i)-0.08*qRange(i),qMax(i)+0.08*qRange(i),numNodes)';
        
        chainSorted = sort(qChain(:,i));
        i1 = floor((numIterations+1)/4);
        i3 = floor((numIterations+1)/4*3);
        f1 = (numIterations+1)/4-i1;
        f3 = (numIterations+1)/4*3-i3;
        q1 = (1-f1)*chainSorted(i1,:)+f1*chainSorted(i1+1,:);
        q3 = (1-f3)*chainSorted(i3,:)+f3*chainSorted(i3+1,:);
        iRange = q3-q1;
    
        if iRange<=0
            s = 1.06*std(qChain(:,i))*numIterations^(-1/5);
        else
            s = 1.06*min(std(qChain(:,i)),iRange/1.34)*numIterations^(-1/5);
        end
    
        for j = 1:1:numNodes
            err = (qVals(j,i)-qChain(:,i))/s;
            qProbs(j,i) = 1/numIterations*sum(exp(-0.5*err.^2)/sqrt(2*pi))./s;
        end
    end
    
    % Save the results.
    densities.qVals = qVals;
    densities.qProbs = qProbs;
    save('densities.mat','densities');
    
    disp('Posterior densities generated.');
    disp('--------------------------------------------------');
end