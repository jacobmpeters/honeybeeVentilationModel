function outputStructure = simulateVentilationRobinBC(varargin)
% simulateVentilation.m
%
% written by:   Jacob Peters
% contributed:  Orit Peleg
% model by:     Jacob Peters and L. Mahadevan
% date written: May 11, 2015
% last updated: July 5, 2017
%
% FORM: outputStructure = simulateVentilationRobinBC(plotMode,vargin)
%
% Optional input:
%
%'plotMode' is categorical variable controlling how data is visualized
%   0 -> no visualization
%   1 -> visualize rho, T, and V in real time & plot k_on and k_off
%   2 -> visualize using imagesc plots & plot k_on and k_off
%   Defalt: 0
%
%'T_amb' is the ambient temperature (degrees Celsius)
%   Default: 25
%
%'L_x' is the nest entrance size (m)
%   Default: 0.38
%
%'saveInterval' is number of frames between saved frames. Only needed
%   if plotMode is 2 or 3 (i.e., data is being saved).
%   Default: 1
%
%'Dv' is the effective diffusion coefficient for velocity
%   Default: 1E-4;
%
%'Dt' is the diffusion coefficient for temperature
%   Default: 1E-5;
%
%'m' is parameter that sets slope of behavioral switch function.
%   Default: 0.1;
%
%'c' is constant in cooling/heating equation
%   Defalut: 0.05;
%
%'nSteps' is the number of time steps in simulation
%   Default: 50000
%
%'wantClusterAnalysis' indicates whether clustering analysis is wanted
%   0 -> don't run cluster analysis
%   1 -> run cluster analysis
%   Default: 0
%
%'alpha' is a coefficient used in the implementation of diffusion of T at 
% the boundaries of the nest entrance (i.e., hive wall).
%   0 -> boundary is perfect conductor (i.e., gradient is T - T_amb
%   1 -> " perfect insulator (i.e., gradient vanishes at boundary)
%   0<alpha<1 -> " somewhere in between
%   Default: 0.9
%
%'seed' is an integer used to seed the random number generator in prior to
%   the simulation. Set seed to 0 to generate random seed and make
%   simulations unique. If a positive integer is used (e.g., 1,2,3,..)
%   simulations will be reproducible.
%   0 <- unique simulation 
%   seed>0 <- reproducible simulation
%   Default: 1
%
% Outputs:
%
%*Note: All outputs are stored in struct called outputStructure. Output
% variables listed are fields within outputStructure.
%
%rho: local fanner density. rho is an mxn matrix where m is the number of
%   time steps and n is the number of x positions.
%
%V: local velocity. V is an mxn matrix where m is the number of time
%   steps and n is the number of x positions.
%
%T: local temperature. T is an mxn matrix where m is the number of time
%   steps and n is the number of x positions.
%
%timeArray: array with data for time axis. Needed because data for every
%   time step may not be output depending on value of plotMode.
%
%optionalInput: a cell array holding values of manually supplied inputs
%
%parameters: a structure holding values for all model parameters
%
%seed: data provided by matlab about the random number generator using rng
%   function. Can be used to replicate simulation.

set(0,'defaultAxesFontName','cmr12','DefaultAxesFontSize',12)

%% Extract optional inputs or set to defaults
optionalInput = varargin;
[plotMode,T_amb,L_x,saveInterval,Dv,Dt,m,...
    c,nSteps,clusterAnalysisWanted,alpha,seed] = ...
    extractOptionalInputs(optionalInput);

%% Set other variables
v_b = 1;            % velocity produced by individual bee
l_b = 0.02;         % wingspan of bee (m)
rho_max = 10/l_b;   % maximum density at given position along nest entrance
T_hive = 36;        % hive temperature (constant)
onSetPoint = T_hive; % prob. of beginning to fan at this Temp. is 0.5
offSetPoint = T_hive; % prob. of ceasing to fan at this Temp. is 0.5 
nBins = L_x/l_b;     % length of entrance/x-axis in bee widths

% seed random number generator
rng(seed)

parameters = struct('v_b',v_b,'l_b',l_b,'rho_max',rho_max,...
    'T_hive',T_hive,'onSetPoint',onSetPoint,'offSetPoint',offSetPoint,...
    'T_amb',T_amb,'saveInterval',saveInterval,'L_x',L_x,'Dv',Dv,...
    'Dt',Dt,'m',m,'c',c,'nSteps',nSteps,'alpha',alpha,'nBins',nBins);

outputStructure.optionalInput = optionalInput;
outputStructure.parameters = parameters;
outputStructure.seed = seed;

%% Define k_on and k_off (sigmoidal functions)
% Continuous notation:
%
% $k_{on} = \frac{\tanh(m(T-T_{on}))+1}{2}$;
% $k_{off} = -\frac{\tanh(m(T-T_{off}))+1}{2}$
%

span = 50;
increment = 0.1;
T_x = T_hive-span:increment:T_hive+span; % local temperature

% probabality of beginning to fan as a function of local temperature, T
k_on = (tanh(m.*(T_x-onSetPoint))+1)/2;

% probabality of ceasing to fan as a function of local temperature, T
k_off = flip((tanh(m.*(T_x-offSetPoint))+1)/2);

%% Plot probability of k_on and k_off

if plotMode==1 || plotMode==2 % check if user wants to plot k_on & k_off
    figure('color','w')
    plot(T_x,k_on,'g-','LineWidth',2)  % plot k_on
    hold on
    plot(T_x,k_off,'r-','LineWidth',2) % plot k_off
    % plot vertical line at onSetPoint
    plot([T_hive,T_hive],[0,1],'k-.','LineWidth',2)
    
    % check if setpoints are the same, plot accordingly
    T_hive_equals_T_amb = isequal(T_hive,T_amb);
    if T_hive_equals_T_amb
        legend({'k_{on}','k_{off}','T_{hive} & T_{amb}'})
    else
        plot([T_amb,T_amb],[0,1],'b--','LineWidth',2)
        legend({'k_{on}','k_{off}','T_{hive}','T_{amb}'})
    end
    xlim([T_hive-30 T_hive+30])
    hold off
    % specify plot properties
    xlabel('Local Temperature')
    ylabel('Probability')
    grid on
end

%% Set initial conditions for $\rho$, $V$, & $T$
% 'history' structure will be used to store data arrays, if saveInterval...
% is not 1, not all of history will be saved to output file.

% density is 0 at every position
%history(1).rho = zeros(1,nBins);  

% rho array is random initionally
%rand_rho_array = randi(10,1,nBins-2)/l_b; 
%history(1).rho = horzcat(0,rand_rho_array,0);

% every other postion has bees
%history(1).rho = horzcat(0,repmat([50 100],1,8),50,0);

% uniform distribution
%history(1).rho = horzcat(0,repmat(100,1,17),0); 
history(1).rho = horzcat(0,repmat(50,1,17),0); 


history(1).V   = zeros(1,nBins);  % velocity is 0 at every position
history(1).T   = ones(1,nBins)*T_hive; % set T to T_hive everywhere

outputStructure(1).rho = history(1).rho;
outputStructure(1).V   = history(1).V;
outputStructure(1).T   = history(1).T; 


if plotMode ~= 0
    disp(history(1))
end

%% Begin simulation:
save_counter = 1; % initialize counter
structure_counter = 1; % initialize counter

for i = 2:nSteps
    
    %% randomly select x position to update
    IMIN = 2; % don't allow fanning at first position (boundary)
    IMAX = nBins-1;% don't allow fanning at penultimate position (boundary)
    indx = randi([IMIN,IMAX]); % randomly select x position to update
    % +1 is used to ignore first bin
    rho_tmp = history(i-1).rho(indx); % get current density at position
    %V_tmp   = history(i-1).V(indx);   % get current velocity at position
    T_tmp = history(i-1).T(indx);   % get current temp at position
    
    %% Equation 1: Bees decide to fan or stop fanning given local temp.
    %
    % $$\frac{\partial\rho(x,t)}{\partial t} =
    % \left[k_{\mathrm{on}}\left(T(x,t)\right) -
    % k_{\mathrm{off}}\left(T(x,t)\right)\right],
    % \quad \rho(x,t) \in [0,\rho_\mathrm{max}]$$
    %
    % determine probability of turning on or off given local temperature
    
    % find closest value of T used in K_on function
    temperature_diff = abs(T_x - T_tmp);
    indx_T = find(temperature_diff==min(temperature_diff),1,'first');
    prob_on  = k_on(indx_T);  % probability that bee will fan
    prob_off = k_off(indx_T); % probability that bee will stop fanning
    
    % determine whether bee turns off
    if rho_tmp >= 1/l_b % check if any bees fanning at postion
        n_off = rand<prob_off; % determine if it should stop
    else
        n_off = 0;
    end
    
    % determine whether bee turns on
    if rho_max-rho_tmp > 0 % check if bees available to fan
        n_on = rand<prob_on;
    else
        n_on = 0;
    end
    
    rho_new = rho_tmp + (n_on - n_off)/l_b; % update local density
    
    % enforce maximum density stipulation (shouldn't be necessary)
    if rho_new > rho_max
        rho_new = rho_max;
    end
    
    
    
    current_rho_array = history(i-1).rho;
    current_rho_array(indx) = rho_new; % replace old local rho with new

    %% Equation 2: compute local velocities given distrubution of fanners
    %
    % $$v(x,t) = l_b v_b  \left[   \rho(x,t) - \frac{1}{L_x} \int_0^{L_x}
    % \rho(x,t) dx  \right] + D_{v} \frac{\partial ^2v(x,t)}{\partial
    % x^2}$$
    %
    % breakdown for computation:
    %
    % $$v(x,t) = P\A$$
    %
    % $$A = l_b v_b B$$
    %
    % $$B = \left[ \rho(x,t) - \frac{1}{L_x} \int_0^{L_x} \rho(x,t) dx
    % \right]$$
    
    % compute B considering only interior bins where bees can fan
    B = current_rho_array(2:end-1) - sum(current_rho_array(2:end-1))*...
        l_b/((nBins-2)*l_b);
    B = horzcat(0,B,0);
    A = v_b*l_b*B;

    % build matrix with diagonal (f1) and off diagonal (f2) formulas
    delta = l_b;
    
    f1 = 1 + 2*Dv/(delta^2);
    f2 = -Dv/(delta^2);
    
    M1 = diag(ones(1,nBins-1),1);
    M2 = diag(ones(1,nBins),0);
    M3 = diag(ones(1,nBins-1),-1);
    P = M1*f2+M2*f1+M3*f2;
    
    % solve for V (called V_new_array for legacy compatability)
    V = P\A';
    V_new_array = V';
    
%     %% Old (wrong way)
%
%     % compute 2nd derivative of v(x,t)
%     d2v = zeros(1,nBins)
%
%     % implement for middle bins
%     for kk = 2:nBins-1
%         d2v(kk) = A(kk+1) + A(kk-1) - 2*A(kk);
%     end
%     
%     % complete 2nd derivative calculations by dividing by dx^2
%     dx2 = l_b^2;
%     %dx2 = 1;
%     d2v_dx2 = d2v/dx2;
%  
%     % calculate C
%     C = Dv.*d2v_dx2;
%     
%     % compute new velocity array
%     V_new_array = A + C;
%     
    %% Equation 3:  update temperature given new local velocities (FIX THIS!)
    %
    % $$\frac{\partial T(x,t)}{\partial t} = -c v(x,t) \Delta T$$
    %
    % if $v \geq 0$, $\Delta T = T_{h} - T$
    %
    % if $v < 0$, $\Delta T = T - T_{a}$
    
    previous_array = [history(i-1).T];
    delta_Ta_array = previous_array - T_amb;
    delta_Th_array = T_hive - previous_array;
    
    % initialize
    T_array = previous_array;

    for j = 2:nBins % start at 2 because velocity is zero at wall
        if V_new_array(j) > 0 
            change_in_T = c*V_new_array(j)*delta_Th_array(j);
        elseif V_new_array(j) < 0
            change_in_T = c*V_new_array(j)*delta_Ta_array(j);
        elseif V_new_array(j) == 0
            change_in_T = 0;
        end
        T_array(j) = previous_array(j) + change_in_T;
    end
    
    %% Add diffusion to smooth temperature profile
    %
    % $$T(x,t) = T_{t-1}(x) + \frac{\partial T(x,t)}{\partial t} +
    % D_{T} \frac{\partial ^2 T(x,t)}{\partial x^2}$$
    
    % Initialize array and add two extra bins for ghost points and the wall
    % wall points
    d2T = zeros(1,nBins); 
    
    % Define ghost point on either side for implementation of 
    % (mixed) Robin boundary condition.

    T_ghost_beg =...
        T_array(2) - (2*(1-alpha)*l_b/alpha)*(T_array(1)-T_amb);
    T_ghost_end =...
        T_array(end-1) - (2*(1-alpha)*l_b/alpha)*(T_array(end)-T_amb);
    
    % implement diffusion for ends of array first (using ghost points)
    d2T(1) = T_ghost_beg - 2*T_array(1) + T_array(2) ;
    d2T(end) = T_ghost_end  - 2*T_array(end) + T_array(end-1);
  
    % then implement for internal bins
    for kk = 2:nBins-1
        d2T(kk) = T_array(kk-1) - 2*T_array(kk) + T_array(kk+1) ;
    end
    
    % complete second derrivative calculation by deviding by dx2
    dx2 = l_b^2; 
    %dx2 = 1;
    d2T_dx2 = d2T/dx2;
    
    
    T_new_array = T_array + Dt.*d2T_dx2; 
    
    %% keep a record
    history(i).rho = current_rho_array; 
    history(i).V   = V_new_array;       
    history(i).T   = T_new_array;    
    
    %% plot $\rho$, $V$, & $T$ in real time
    if plotMode == 1 % check if user wants to plot in real time
          % Dimentioned figures
%         subplot(3,1,1)
%         plot([history(i).rho]); box off
%         ylabel('density')
%         subplot(3,1,2)
%         plot([history(i).V]); box off
%         ylabel('velocity');
%         subplot(3,1,3)
%         plot([history(i).T]); box off
%         ylabel('temperature (C)')
%         xlabel('x-position')
%         pause(0.01)
        
        % Dimentionless figures
        subplot(3,1,1)
        plot([history(i).rho].*l_b); 
        box off
        ylabel('$\varphi = \rho l_b$','interpreter','latex')
        subplot(3,1,2)
        plot([history(i).V]./v_b); 
        box off
        ylabel('$u = \frac{V}{v_{b}}$','interpreter','latex');
        subplot(3,1,3)
        plot([history(i).T]); 
        box off
        ylabel('$T \left( ^\circ C \right)$','interpreter','latex')
        xlabel('$\hat{x} = \frac{x}{l_b}$','interpreter','latex')
        %pause(0.01)
        pause(0.05)
    end
    
    %% compute moments
    [coefficientOfVariation,firstMoment,secondMoment,thirdMoment] =...
        compute_moments(current_rho_array);
    %% compute flux
    flux = compute_flux(V_new_array,L_x);
    %% compute cost of friction 
    costOfFriction =  compute_costOfFriction(V_new_array,l_b);
    
    %% compute normalized friction
    % normalize velocity data
    V = history(i).V;
    normalizedV = ...
        (V-min(V))/(max(V)-min(V));
    normCostOfFriction = compute_costOfFriction(normalizedV,l_b);
    
    %% save data at specified time intervals (indicated by saveInterval)
    if save_counter == saveInterval
        % add data arrays to saved structure
        save_counter = 0;
        structure_counter = structure_counter + 1;
        
        outputStructure(structure_counter).rho = history(i).rho;
        outputStructure(structure_counter).V = history(i).V;
        outputStructure(structure_counter).normalizedV = normalizedV;
        outputStructure(structure_counter).T = history(i).T;
        %outputStructure(structure_counter).V_diffusion_array = C;
        outputStructure(structure_counter).timeArray = i;
        outputStructure(structure_counter).firstMoment = firstMoment;
        outputStructure(structure_counter).secondMoment = secondMoment;
        outputStructure(structure_counter).thirdMoment = thirdMoment;
        outputStructure(structure_counter).coefficientOfVariation = ...
            coefficientOfVariation;
        outputStructure(structure_counter).costOfFriction = costOfFriction;
        outputStructure(structure_counter).flux = flux;
        outputStructure(structure_counter).normCostOfFriction = ...
            normCostOfFriction;
        outputStructure(structure_counter).numFanners = ...
            sum([history(i).rho]);
        
        % run cluster analysis if wanted
        if clusterAnalysisWanted == 1
            outputStructure(structure_counter).order_parameter = ...
                clusterAnalysis(current_rho_array,rho_max,...
                nBins,nSteps,l_b);
        end
    end
    
    save_counter = save_counter + 1;
    
end
%% plot images $\rho$, $V$, & $T$ at end of simulation
if plotMode == 2
    
    rhoMatrix = reshape([outputStructure.rho],nBins,nSteps/saveInterval);
    tMatrix = reshape([outputStructure.T],nBins,nSteps/saveInterval);
    vMatrix = reshape([outputStructure.V],nBins,nSteps/saveInterval);
      
      % Dimentioned figures
%     figure('color','w')
%     subplot(3,1,1); imagesc(rhoMatrix); 
%     title('\rho (bees/m)'); 
%     colorbar
%     subplot(3,1,2); imagesc(tMatrix); 
%     cstr = sprintf('T (%c)', char(176));
%     title(cstr); 
%     colorbar
%     subplot(3,1,3); 
%     imagesc(vMatrix); 
%     title('V (m/s)'); 
%     colorbar

    % Dimentionless figures
    figure('color','w','Position',[450 200 550 600]);
    
    for i = 1:3
        ax(i) = subplot(3,1,i);
    end
    linkaxes(ax,'x')
    
    subplot(ax(1))
    imagesc(rhoMatrix.*l_b);
    ylabel('$\hat{x}$','interpreter','latex','FontSize',18)
    xlabel('$\hat{t}$','interpreter','latex','FontSize',18)
    title('$\varphi = \rho l_b$','interpreter','latex','FontSize',20); 
    colorbar('peer',ax(1))
    
    subplot(ax(3))
    imagesc(tMatrix/T_hive);
    ylabel('$\hat{x}$','interpreter','latex','FontSize',18)
    xlabel('$\hat{t}$','interpreter','latex','FontSize',18)
    %title('$T \left( ^\circ C \right)$','interpreter','latex');
    title('$\Theta = T/T_{hive}$','interpreter','latex','FontSize',20);
    colorbar('peer',ax(3))
     
    subplot(ax(2))
    imagesc(vMatrix./v_b); 
    ylabel('$\hat{x}$','interpreter','latex','FontSize',18)
    xlabel('$\hat{t}$','interpreter','latex','FontSize',18)
    title('$u = V/v_b$','interpreter','latex','FontSize',20); 
    colorbar('peer',ax(2))
    
    %%%
    figure
    
    subplot(5,1,1)
    COF = [outputStructure(2:end).costOfFriction];
    x = 1:length(COF);
    plot(x,COF)
    ylabel('$\int \mu\left(\frac{dV}{dx}\right)^2dx$',...
        'interpreter','latex','FontSize',16) 
    xlabel('$\hat{t}$','interpreter','latex','FontSize',16)
    
    subplot(5,1,2)
    secondMoment = [outputStructure(2:end).secondMoment];
    x = 1:length(secondMoment);
    plot(x,secondMoment)
    ylabel('2nd Moment','FontSize',16) 
    xlabel('$\hat{t}$','interpreter','latex','FontSize',16)
    
    subplot(5,1,3)
    flux = [outputStructure(2:end).flux];
    plot(x,flux)
    ylabel('flux','FontSize',16) 
    xlabel('$\hat{t}$','interpreter','latex','FontSize',16)
    
    subplot(5,1,4)
    normCostOfFriction = [outputStructure(2:end).normCostOfFriction];
    plot(x,normCostOfFriction)
    ylabel('normCostOfFriction','FontSize',16) 
    xlabel('$\hat{t}$','interpreter','latex','FontSize',16)
    
    subplot(5,1,5)
    flux_per_friction = flux./COF;
    plot(x,flux_per_friction)
    ylabel('flux/friction','FontSize',16) 
    xlabel('$\hat{t}$','interpreter','latex','FontSize',16)
    
    %%%
    
end

end

%% function for computing moments of density distribution
function [coefficientOfVariation,firstMoment,secondMoment,thirdMoment] =...
    compute_moments(rho)
    x  = linspace(-1,1,length(rho)); % force position to range from -1 to 1
    firstMoment = sum(rho.*x)/sum(rho);
    secondMoment = sum(rho.*(x-firstMoment).^2)/sum(rho);
    thirdMoment = sum(rho.*(x.^3))/sum(rho);
    coefficientOfVariation = abs(firstMoment)/secondMoment;
end

%% function for computing cost of friction
function costOfFriction =  compute_costOfFriction(v,dx)
    %u = 1.86E-5; % viscosity of air at 25C (kg/(m*s))
    %costOfFriction = sum(u*(diff(v)/dx).^2)*dx; 
    dx = 1;
    costOfFriction = sum((diff(v)/dx).^2)*dx; 
end

%% function for computing flux of air through entranc
function flux = compute_flux(V_new_array,L_x)
    flux = (sum(abs(V_new_array))*L_x)/2;
end

%% function for running cluster analysis
function order_parameter = clusterAnalysis(curr_rho_array,rho_max,...
    nBins,nSteps,l_b)
rho_max = rho_max*l_b;
Nc = 0;
N_cluster = 0;
for pos_i = 1:nBins
    switch pos_i
        case 1
            if curr_rho_array(pos_i)>0
                % create first cluster
                Nc = Nc+1;
                clusters_ID{Nc} = pos_i;
            end
        otherwise
            if curr_rho_array(pos_i)>0 && curr_rho_array(pos_i-1)>0
                clusters_ID{Nc} = [clusters_ID{Nc}, pos_i];
            elseif curr_rho_array(pos_i)>0
                Nc = Nc+1;
                clusters_ID{Nc} = pos_i;
            end
    end
end

if Nc>0
    Rc = zeros(1, Nc);
    Non = 0;
    for cluster_i = 1:Nc
        curr_elements_in_cluster = clusters_ID{cluster_i};
        if isempty(curr_elements_in_cluster)==0
            Rc(cluster_i) = length(curr_elements_in_cluster);
            Non = Non+sum(curr_rho_array(curr_elements_in_cluster));
        else
            Rc(cluster_i) = NaN;
        end
    end
    mean_Rc = nanmean(Rc);
else
    mean_Rc = 0;
    Non = 0;
    Nc=0;
end

N_max = nBins*rho_max;
R_max = nBins;
order_parameter = (Non/N_max)*(mean_Rc/R_max);
clear clusters_ID

end

%% function called at begginning of script to read optional inputs
function [plotMode,T_amb,L_x,saveInterval,Dv,Dt,m,c,...
    nSteps,clusterAnalysisWanted,alpha,seed] =...
    extractOptionalInputs(optionalInput)

% check for optional input 'plotMode'
indxP = strcmp('plotMode', optionalInput);
if sum(indxP) == 0 % if no value provided, use default
    plotMode = 2;
else % if value provided, define plotMode using input value
    plotMode = cell2mat(optionalInput(find(indxP==1) + 1));
end
clear indxP

% check for optional input 'T_amb'
indxT = strcmp('T_amb', optionalInput);
if sum(indxT) == 0 % if no value provided, use default
    T_amb = 32;
else % if value provided, define T_amb using input value
    T_amb = cell2mat(optionalInput(find(indxT==1) + 1));
end
clear indxT

% check for optional input 'L_x'
indxL = strcmp('L_x', optionalInput);
if sum(indxL) == 0
    L_x = 0.38;
else
    L_x = cell2mat(optionalInput(find(indxL==1) + 1));
end
clear indxL

% check for optional input 'saveInterval'
indxS = strcmp('saveInterval', optionalInput);
if sum(indxS) == 0
    saveInterval = 1;
else
    saveInterval = cell2mat(optionalInput(find(indxS==1) + 1));
end
clear indxS

% check for optional input 'Dv'
indxDv = strcmp('Dv', optionalInput);
if sum(indxDv) == 0
    %Dv = 10E-4;
    Dv = 1E-3;
else
    Dv = cell2mat(optionalInput(find(indxDv==1) + 1));
end
clear indxDv

% check for optional input 'Dv'
indxDt = strcmp('Dt', optionalInput);
if sum(indxDt) == 0
    Dt = 5E-5;
else
    Dt = cell2mat(optionalInput(find(indxDt==1) + 1));
end
clear indxDt


% check for optional input 'm'
indxM = strcmp('m', optionalInput);
if sum(indxM) == 0
    m = 0.1;
else
    m = cell2mat(optionalInput(find(indxM==1) + 1));
end
clear indxM

% check for optional input 'c'
indxC = strcmp('c', optionalInput);
if sum(indxC) == 0
    c = 0.05;
else
    c = cell2mat(optionalInput(find(indxC==1) + 1));
end
clear indxC

% check for optional input 'nSteps'
indxN = strcmp('nSteps', optionalInput);
if sum(indxN) == 0
    nSteps = 2000;
else
    nSteps = cell2mat(optionalInput(find(indxN==1) + 1));
end
clear indxN

% check for optional input 'wantClusterAnalysis'
indxW = strcmp('clusterAnalysisWanted', optionalInput);
if sum(indxW) == 0
    clusterAnalysisWanted = 0;
else
    clusterAnalysisWanted = cell2mat(optionalInput(find(indxW==1) + 1));
end
clear indxW

% check for optional input 'alpha'
indxA = strcmp('alpha', optionalInput);
if sum(indxA) == 0
    alpha = 0.25;
else
    alpha = cell2mat(optionalInput(find(indxA==1) + 1));
end
clear indxA

% check for optional input 'seed'
indxS = strcmp('seed', optionalInput);
if sum(indxS) == 0
    seed = 1;
else
    seed = cell2mat(optionalInput(find(indxS==1) + 1));
    if seed == 0
        seed = rand;
    end
end
clear indxS

end
