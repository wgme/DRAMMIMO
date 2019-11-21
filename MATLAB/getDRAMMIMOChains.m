%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By: Wei Gao (wg14@my.fsu.edu)
% Last Modified: 11/20/2019
% Desciption:
% 1. Based on the code from Dr. Marko Laine 
%    (http://helios.fmi.fi/~lainema/mcmc/).
% 2. Also based on the math from Dr. Ralph C. Smith 
%    (Uncertainty Quantification: Theory, Implementation, and Applications).
% V01: The code was used for SMASIS 2017, with only MATLAB version. 
%      The code had SISO, SIMO and MIMO subversions initially, and they 
%      were later unified to the MIMO version.
%      The code was only generating estimation chains.
% V02a: The code was used for SMASIS 2018, with Python version added and 
%       MATLAB version modified to be the same.  
%       The code was only generating estimation chains.
% V02b: The code was used for dissertation, with both MATLAB and Python 
%       versions. The code for generating credible and prediction intervals
%       was added.
% V03: The code was rearranged to have three major components for both 
%      MATLAB and Python versions: getDRAMMIMOChains(), 
%      getDRAMMIMODensities(), and getDRAMMIMOIntervals().
% V04: An output "prior" was added to getDRAMMIMOChains(), containing 
%      parameters for inverse-wishart sampling.  
%      Parameters for displaying and saving results were added to DRAMParams.
%      Some minor tweaks.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data: cell array, size = 1 * N. 
%       Each cell is a vector, size = n * 1.
%       N = number of data sets (N=1 for Bayesian, N>1 for Max Entropy).
%       n = number of observation points.
% model: struct.
%        .fun, cell array, size = 1 * N. Each cell is a function handle.
%        .errFun, cell array, size = 1 * N. Each cell is a function handle.
% modelParams: struct.
%              .names, cell array, size = 1 * p. Each cell is a string.
%              .values, cell array, size = 1 * p. Each cell is a scalar.
%              .lowerLimits, cell array, size = 1 * p. Each cell is a scalar.
%              .upperLimits, cell array, size = 1 * p. Each cell is a scalar.
%              .extra, cell array, size = 1 * p. Each cell is a cell array.
%              p = number of parameters to be estimated.
% DRAMParams: struct.
%             .numIterationsDone, scalar.
%             .numIterationsExpected, scalar.
%             .numIterationsDisplay, scalar.
%             .numIterationsSave, scalar.
%             .previousResults.prior.psi_s, matrix, N * N.
%             .previousResults.prior.nu_s, scalar.
%             .previousResults.chain_q, matrix, Mo * p.
%             .previousResults.last_cov_q, matrix, p * p.
%             .previousResults.chain_cov_err, matrix, N * N * Mo.
%             Mo = number of iterations already done.
% prior: struct.
%        .psi_s, matrix, N * N.
%        .nu_s, scalar.
% chain_q: matrix, M * p.
%          M = number of iterations expected.
% last_cov_q: matrix, p * p.
% chain_cov_err: matrix, N * N * M.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [prior,chain_q,last_cov_q,chain_cov_err] = getDRAMMIMOChains(data,model,modelParams,DRAMParams)
    disp('--------------------------------------------------');
    disp('Initializing estimation chains...');
    
    %% Initialize the parameters.
    
    % Number of data sets.
    N = length(data.xdata);
    % Number of data within each set.             
    n = size(data.xdata{1},1);           
    % Number of model parameters for estimation.
    p = length(modelParams.values);          
    % Number of estimation iterations already done.
    Mo = DRAMParams.numIterationsDone;       
    % Number of estimation iterations to be done in total.  
    M = DRAMParams.numIterationsExpected;
    
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
        cov_err = err'*err;
    else
        cov_err = DRAMParams.previousResults.chain_cov_err(:,:,end);
    end
    cov_err_inv = cov_err\eye(size(cov_err));
    
    % Parameters for sampling random covariance matrix of model prediction errors from inverse-wishart distribution.
    if isempty(DRAMParams.previousResults.prior.psi_s)
        psi_s = zeros(N,N);
        nu_s = 0;
    else
        psi_s = DRAMParams.previousResults.prior.psi_s;
        if isempty(DRAMParams.previousResults.prior.nu_s)
            nu_s = 1;
        else
            nu_s = DRAMParams.previousResults.prior.nu_s;
        end
    end
    prior.psi_s = psi_s;
    prior.nu_s = nu_s;
    
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
    
    %% Initialize the chains.
    
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
    
    disp('Generating estimation chains...');
    disp('--------------------------------------------------');
    
    %% Generate the chains.
    
    for k = Mo+1:1:M
        
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
                [randomWalk,flag] = chol(cov_q);
                % In case the cholesky decomposition failed...
                if flag
                    [randomWalk,flag] = chol(cov_q+1E-8*eye(size(cov_q)));
                    disp('Parameter covariance matrix got adjusted because of singularity.');
                    if flag
                        error('Singular parameter covariance matrix. No adapt.');
                    end
                end
                R1 = randomWalk*sp;
                R2 = randomWalk*sp/5;
            end
        end
        
        %%%%%%%% End of Adaptive Metropolis %%%%%%%%     
        
        % Display current model parameter estimations every xth iteration.
        % Modify the number in mod() after k as needed.
        % Comment this out if unnecessary, i.e. to avoid time delay.
        if mod(k, DRAMParams.numIterationsDisplay) == 0
            disp('Iterations | Parameter Values');
            disp(strcat(num2str(k, '%010d'), ' | ', num2str(q', '%16.8f')));
            disp('--------------------------------------------------');
        end
        
        % Save current estimation chain every yth iteration.
        % Modify the number in mod() after k as needed.
        % Comment this out if not necessary, i.e. to avoid time delay.
        if mod(k, DRAMParams.numIterationsSave) == 0
            chains.prior = prior;
            chains.chain_q = chain_q;
            chains.last_cov_q = last_cov_q;
            chains.chain_cov_err = chain_cov_err;
            save(strcat('chains', num2str(k), '.mat'), 'chains');
        end
        
    end
    
    % Save the results.
    chains.prior = prior;
    chains.chain_q = chain_q;
    chains.last_cov_q = last_cov_q;
    chains.chain_cov_err = chain_cov_err;
    save('chains.mat','chains');
    
    disp('Estimation chains generated.');
    disp('--------------------------------------------------');
end
