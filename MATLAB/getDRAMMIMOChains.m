%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By: Wei Gao (wg14@my.fsu.edu)
% Last Modified: 07/24/2019
% Desciption:
% 1. Based on the code from Dr. Marko Laine 
%    (http://helios.fmi.fi/~lainema/mcmc/).
% 2. Also based on the math from Dr. Ralph C. Smith 
%    (Uncertainty Quantification: Theory, Implementation, and Applications).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [chain_q,last_cov_q,chain_cov_err] = getDRAMMIMOChains(data,model,modelParams,DRAMParams)
    %% Initialize the parameters.
    
    % Number of data sets.
    N = length(data.xdata);
    % Number of data within each set.             
    n = size(data.xdata{1},1);           
    % Number of model parameters for estimation.
    p = length(modelParams.values);          
    % Number of estimation iterations already done.
    Mo = DRAMParams.numDRAMIterationsDone;       
    % Number of estimation iterations to be done in total.  
    M = DRAMParams.numDRAMIterations;
    
    % Best model parameter estimation.
    if Mo==1
        q = [modelParams.values{:}]';
    else
        q = DRAMParams.previousResults.chain_q(end,:)';
    end
    % Old model parameter estimation.
    q0 = zeros(size(q));
    % 1st-stage new model parameter estimation.
    q1 = zeros(size(q));
    % 2nd-stage new model parameter estimation.
    q2 = zeros(size(q));
    
    % Model prediction errors caused by q.
    err = zeros(n,N);                       
    for i=1:1:N
        err(:,i) = model.errFun{i}(q,data.xdata{i},data.ydata{i},modelParams.extra{i});   
    end
    % Model prediction errors caused by q0.
    err0 = zeros(n,N);                      
    % Model prediction errors caused by q1.
    err1 = zeros(n,N);                      
    % Model prediction errors caused by q2.
    err2 = zeros(n,N);                      
    
    % Initialize the covariance matrix of model parameter estimations and its inverse.
    if Mo==1
        cov_q = diag((q~=0).*(0.05*q).^2+(q==0).*1.0);
    else
        cov_q = DRAMParams.previousResults.last_cov_q;
    end
    cov_q_inv = cov_q\eye(size(cov_q));
    
    % Initialize the covariance matrix of model prediction errors and its inverse.
    if Mo==1
        cov_err = diag(repmat(1e-4,1,N));
    else
        cov_err = DRAMParams.previousResults.chain_cov_err(:,:,end);
    end
    cov_err_inv = cov_err\eye(size(cov_err));
    
    % Parameters for sampling random covariance matrix of model prediction errors from inverse-wishart distribution.
    psi_s = diag(repmat(1e-4,1,N));
    nu_s = 1;
    
    % Parameters for Adaptive Metropolis.
    % Adaptive interval.
    ko = 100;
    % Adaptive scale.
    sp = 2.38/sqrt(p);
    % Current mean parameter estimations.
    if Mo==1
        qBar = q;
    else
        qBar = mean(DRAMParams.previousResults.chain_q,1)';
    end
    % Current covariance matrix of parameter estimations.
    qCov = cov_q;
    
    % Parameters for Delayed Rejection.
    % Maximum random walk step size.
    randomWalk = chol(cov_q);
    % 1st-stage random walk maximum step size.
    R1 = randomWalk;
    % 2nd-stage random walk maximum step size.
    R2 = randomWalk/5;
    
    %% Initialize the chain.
    
    % The chain of model parameter estimations for posterior densities.
    chain_q = zeros(M,p);
    if Mo==1
        chain_q(1,:) = q';
    else
        chain_q(1:Mo,:) = DRAMParams.previousResults.chain_q;
    end
    
    % The chain of model parameter estimation covariances is not of interest for now.
    % Record only the latest value instead.
    last_cov_q = cov_q;
    
    % The chain of model prediction errors is not of interest for now.
    
    % The chain of model prediction error covariances for uncertainty propagation.
    chain_cov_err = zeros(N,N,M);
    if Mo==1
        chain_cov_err(:,:,1) = cov_err;
    else
        chain_cov_err(:,:,1:Mo) = DRAMParams.previousResults.chain_cov_err;
    end
    
    %% Generate the chain.
    for k = Mo+1:1:M
        % Display current model parameter estimations every xth iteration.
        % Modify the number in mod() after k as needed.
        % Comment this out if unnecessary, i.e. to avoid time delay.
        if mod(k,2E2)==0
            [k,q']
        end
        
        %%%%%%%% Start of Delayed Rejection %%%%%%%%
        
        % Record the best guess from last step as the old guess.
        q0 = q;
        err0 = err;
        
        % 1st stage Random Walk.
        q1 = q0+R1'*randn(size(q));
        
        if any(q1<[modelParams.lowerLimits{:}]') || any(q1>[modelParams.upperLimits{:}]')
            % If the new guess is out of the bounds ...
            err1 = inf(n,N);
            SS0 = trace(err0'*err0*cov_err_inv);
            SS1 = inf;
            pi10 = 0;
            alpha10 = min(1,pi10);
        else
            % If the new guess is within the bounds ...
            for i=1:1:N
                err1(:,i) = model.errFun{i}(q1,data.xdata{i},data.ydata{i},modelParams.extra{i});
            end
            
            if any(any(isnan(err1)))
                % If the new guess is causing the model response to be NaN ...
                err1 = inf(n,N);
                SS0 = trace(err0'*err0*cov_err_inv);
                SS1 = inf;
                pi10 = 0;
                alpha10 = min(1,pi10);
            else
                % If the new guess is okay ...
                SS0 = trace(err0'*err0*cov_err_inv);
                SS1 = trace(err1'*err1*cov_err_inv);
                % pi(q1|q0)
                pi10 = exp(-0.5*(SS1-SS0));	
                % alpha(q1|q0)
                alpha10 = min(1,pi10);	
            end
        end
        
        % Decide whether to accept the 1st stage new guess.
        if alpha10>rand
            % Accept the 1st stage new guess
            
            % Record the 1st stage new guess as the best guess.
            q = q1;
            err = err1;
            
        else
            % Reject the 1st stage new guess.
            
            % 2nd stage Random Walk.
            q2 = q0+R2'*randn(size(q));
            
            if any(q2<[modelParams.lowerLimits{:}]') || any(q2>[modelParams.upperLimits{:}]')
                % If the new guess is out of the bounds ...
                err2 = inf(n,N);
                SS2 = inf;
                pi20 = 0;
                pi12 = 0;
                alpha12 = 0;
                alpha210 = 0;
            else
                % If the new guess is within the bounds ...
                for i=1:1:N
                    err2(:,i) = model.errFun{i}(q2,data.xdata{i},data.ydata{i},modelParams.extra{i});
                end
                
                if any(any(isnan(err2)))
                    % If the new guess is causing the model response to be NaN ...
                    err2 = inf(n,N);
                    SS2 = inf;
                    pi20 = 0;
                    pi12 = 0;
                    alpha12 = 0;
                    alpha210 = 0;
                else
                    % If the new guess is okay ...
                    SS2 = trace(err2'*err2*cov_err_inv);
                    % pi(q1|v)/pi(q0|v)
                    pi20 = exp(-0.5*(SS2-SS0));                 
                    % J(q1|q2)
                    J12 = exp(-0.5*(q1-q2)'*cov_q_inv*(q1-q2)); 
                    % J(q1|q0)
                    J10 = exp(-0.5*(q1-q0)'*cov_q_inv*(q1-q0)); 
                    % pi(q1|v)/pi(q0|v)
                    pi12 = exp(-0.5*(SS1-SS2));                 
                    % alpha(q1|q2)
                    alpha12 = min(1,pi12);                    
                    % alpha(q2|q0,q1)
                    if alpha12==1
                        alpha210 = 0;
                    else
                        alpha210 = min(1,pi20*J12/J10*(1-alpha12)/(1-alpha10));    
                    end
                end
            end
            
            % Decide whether to accept the 2nd stage new guess.
            if alpha210>rand
                % Accept the 2nd stage new guess
                
                % Record the 2nd stage new guess as the best guess.
                q = q2;
                err = err2;
            end
            
        end
        
        %%%%%%%% End of Delayed Rejection %%%%%%%%
        
        %%%%%%%% Start of Adaptive Metropolis %%%%%%%%
        
        % Record the chains.
        chain_q(k,:) = q';
        last_cov_q = cov_q;
        chain_cov_err(:,:,k) = cov_err;
        
        % Update cov_err and cov_err_inv.
        cov_err = iwishrnd(psi_s+err'*err,nu_s+n);
        cov_err_inv = cov_err\eye(size(cov_err));
        
        % Update cov_q and cov_q_inv
        % No update in the 1st round (1 round = ko steps)
        if k==ko
            % Calculate at the end of the 1st round (ko-th step).
            % Mean model parameter estimations.
            qBar = mean(chain_q(1:k,:),1)';
            % Covariance of model parameter estimations.
            qCov = cov(chain_q(1:k,:));
        elseif k>ko
            % Keep calculating after the ko-th step.
            % Mean model parameter estimations.
            qBarOld = qBar;
            qBar = ((k-1)*qBarOld+q)/k;
            % Covariance of model parameter estimations.
            qCovOld = qCov;
            qCov = (k-2)/(k-1)*qCovOld+1/(k-1)*((k-1)*(qBarOld*qBarOld')-k*(qBar*qBar')+q*q');
            
            % Update at the end of every round since the ko-th step.
            if mod(k,ko)==0
                cov_q = qCov;
                cov_q_inv = cov_q\eye(size(cov_q));
                randomWalk = chol(cov_q);
                R1 = randomWalk*sp;
                R2 = randomWalk*sp/5;
            end
        end
        
        %%%%%%%% End of Adaptive Metropolis %%%%%%%%       
    end
end
