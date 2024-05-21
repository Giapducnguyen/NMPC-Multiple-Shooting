% point stabilization + obstacle avoidance + Multiple shooting + Runge Kutta

%{
Source:
https://github.com/MMehrez/MPC-and-MHE-implementation-in-MATLAB-using-Casadi
%}

%{
M o d i f i e d  b y: G i a p  N g u y e n

% Modifications made to the above source:
1 - Adding control rate + SOFT constraints on control rate
   (Reason: Enforcing both HARD control and control rate are not a good practice)
2 - Adding Control horizon # prediction horizon
%}

clear all
close all

import casadi.*

%% Tuning Parameters
% Weights
Q = diag([1, 6, 0.1]);  % on reference tracking
R = diag([0.05, 0.01]);   % on control increment
Ws = diag([1e2, 1e2]);  % on slack variables

h = 0.2;            % sampling time [s]
Np = 10;            % prediction horizon
Nc = 3;             % control horizon
Ns = 2*Nc;          % slack variables over control horizon; 2 == number of controls = {v; omega}

%% Given paramaters
rob_diam = 0.3;     % robot diameter

obs_x = 0.5;        % obstacle x position
obs_y = 0.5;        % obstacle y position
obs_diam = 0.3;     % obstacle diameter

% State constraints
x_max       = 2;          x_min = -x_max;
y_max       = 2;          y_min = -y_max;
theta_max   = Inf;    theta_min = -theta_max;
% Control constraints
v_max       = 0.6;      v_min   = -v_max;
omega_max   = pi/4; omega_min   = -omega_max;
% Control Increment constraints
dv_max      = 0.1;       dv_min = -dv_max;
domega_max  = pi/12; domega_min = -domega_max;

% Boundary of slack variables on Control Increment
Sc_max = [dv_max; domega_max]./ 5;
Sc_min = -Sc_max;

% Symbolic state
x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta');
state = [x;y;theta]; n_state = length(state);

% Symbolic control
v = SX.sym('v'); omega = SX.sym('omega');
control = [v;omega]; n_control = length(control);

% Symbolic continuous system r.h.s
rhs = [v*cos(theta);v*sin(theta);omega]; 

% Symbolic mapping function dxdt = f(x,u)
f = Function('f',{state,control},{rhs}); 

% Parameters = [initial state; reference; previous control]
P = SX.sym('P',n_state + n_state + n_control);

% Decision variable: Optimal states
X = SX.sym('X',n_state,(Np+1));
% Decision variable: Control increments
Delta_U = SX.sym('Delta_U',n_control,Nc); 

% Slack variables over control horizon
S_c = SX.sym('S_c', n_control, Nc);

obj = 0;            % Symbolic objective function value
g_continuity = [];  % continuity constraint vector
g_control = [];     % control constraint vector
g_du = [];          % control increment constraint vector
g_obs = [];         % obstacle avoidance

% % RK4 INTEGRATOR
k1 = f(state, control);
k2 = f(state + h/2*k1, control);
k3 = f(state + h/2*k2, control);
k4 = f(state + h*k3, control);
next_state = state + h/6*(k1 +2*k2 +2*k3 +k4);

RK4 = Function('RK4',{state,control},{next_state});

% initial condition constraints
g_continuity = [g_continuity;X(:,1) - P(1:n_state,1)]; 

% initial obstacle avoidance constraints
g_obs = [g_obs; -sqrt((X(1,1)-obs_x)^2+(X(2,1)-obs_y)^2) + (rob_diam/2 + obs_diam/2)];

for k = 1:Np
    % previous control
    if k == 1
        u_km1 = P(2*n_state+1 : 2*n_state + n_control,1);
    elseif k <= Nc
        du_temp = zeros(n_control,1);
        for j = 1 : k-1
            du_temp = du_temp +  Delta_U(:,j);
        end
        u_km1 = P(2*n_state+1 : 2*n_state + n_control,1) + du_temp;
    else
        u_km1 = P(2*n_state+1 : 2*n_state + n_control,1) + sum(Delta_U,2);
    end
    % current state
    z_k = X(:,k);
    % current control increment, current slack variables
    if k <= Nc
        du_k = Delta_U(:,k);    % current control increment
        sc_k = S_c(:,k);        % current slack variables
    else
        du_k = zeros(2,1);      % current control increment
        sc_k = zeros(2,1);      % current slack variables
    end
    % current reference
    r_k = P(n_state+1 : 2*n_state,1);
    % objective computation
    obj = obj + (z_k-r_k)'*Q*(z_k-r_k) + du_k'*R*du_k + sc_k'*Ws*sc_k;
    % current control = previous control + current control increment
    u_k = u_km1 + du_k;
    % control constraints, control increment constraints
    if k <= Nc
        g_control = [g_control; u_k];
        g_du = [g_du; du_k - sc_k; du_k + sc_k];
    end
    % state propagation
    z_kp1 = RK4(z_k, u_k); 
    % continuity condition (closing the gap)
    x_kp1 = X(:,k+1);
    g_continuity = [g_continuity; x_kp1 - z_kp1];
    % obstacle avoidance
    g_obs = [g_obs; -sqrt((X(1,k+1)-obs_x)^2+(X(2,k+1)-obs_y)^2) + (rob_diam/2 + obs_diam/2)];
end
% Reshape the decision variable to become 1 column vector
OPT_variables = [ reshape(X, n_state*(Np+1),1);
                  reshape(Delta_U, n_control*Nc,1);
                  reshape(S_c, n_control*Nc, 1)     ];

nlp_prob = struct('f', obj, 'x', OPT_variables,...
                    'g', [g_continuity; g_control; g_du; g_obs], 'p', P);

opts                                    = struct;
opts.ipopt.max_iter                     = 2000;
opts.ipopt.print_level                  = 0;%0,3
opts.print_time                         = 0;
opts.ipopt.acceptable_tol               = 1e-8;
opts.ipopt.acceptable_obj_change_tol    = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

% qpoases_opts = struct;
% qpoases_opts.qpsol_options.error_on_fail = false;
% solver = nlpsol('solver', 'sqpmethod', nlp_prob);%,qpoases_opts);

args = struct;

% Continuity constraints (closing the gaps)
args.lbg(1:n_state*(Np+1),1) = 0;  % -1e-20  
args.ubg(1:n_state*(Np+1),1) = 0;  % 1e-20

% Control constraints
args.lbg(n_state*(Np+1)+1 : n_state*(Np+1) + n_control*Nc,1) ...
                                    = kron(ones(Nc,1),[v_min; omega_min]);
args.ubg(n_state*(Np+1)+1 : n_state*(Np+1) + n_control*Nc,1) ...
                                    = kron(ones(Nc,1),[v_max; omega_max]);

% Control Increment constraints with slack variables
args.lbg(n_state*(Np+1)+n_control*Nc+1 : n_state*(Np+1)+n_control*Nc+2*n_control*Nc) ...
    = kron(ones(Nc,1), [-Inf; -Inf; dv_min; domega_min]);
args.ubg(n_state*(Np+1)+n_control*Nc+1 : n_state*(Np+1)+n_control*Nc+2*n_control*Nc) ...
    = kron(ones(Nc,1), [dv_max; domega_max; Inf; Inf]);

% Obstacle avoidance constraints
args.lbg(n_state*(Np+1)+n_control*Nc+2*n_control*Nc+1: ...
        n_state*(Np+1)+n_control*Nc+2*n_control*Nc + Np+1) = -Inf*ones(Np+1,1);
args.ubg(n_state*(Np+1)+n_control*Nc+2*n_control*Nc+1: ...
        n_state*(Np+1)+n_control*Nc+2*n_control*Nc + Np+1) = zeros(Np+1,1);

% State constraints
args.lbx(1:n_state*(Np+1),1) = kron(ones(Np+1,1),[x_min; y_min; theta_min]);
args.ubx(1:n_state*(Np+1),1) = kron(ones(Np+1,1),[x_max; y_max; theta_max]);

% Control Increment Constraints
args.lbx(n_state*(Np+1)+1: n_state*(Np+1) + n_control*Nc,1) = ...
                                    kron(ones(Nc,1),[dv_min; domega_min] + Sc_min);
args.ubx(n_state*(Np+1)+1: n_state*(Np+1) + n_control*Nc,1) = ...
                                    kron(ones(Nc,1),[dv_max; domega_max] + Sc_max);
% Slack variable constraints
args.lbx(n_state*(Np+1)+n_control*Nc+1 : n_state*(Np+1)+n_control*Nc + n_control*Nc) ...
    = kron(ones(Nc,1),Sc_min);
args.ubx(n_state*(Np+1)+n_control*Nc+1 : n_state*(Np+1)+n_control*Nc + n_control*Nc) ...
    = kron(ones(Nc,1),Sc_max);

%----------------------------------------------
% ALL OF THE ABOVE IS JUST A PROBLEM SET UP


% THE SIMULATION LOOP SHOULD START FROM HERE
%-------------------------------------------
t0 = 0;
x0 = [0 ; 0 ; 0.0];     % initial condition.
xs = [1.5 ; 1.5 ; 0.0]; % Reference posture.

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(Nc,2);        % two control inputs for each robot
X0 = repmat(x0,1,Np+1)'; % initialization of the states decision variables

sim_tim = 15; % Maximum simulation time

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];
sc_cl = [];

% the main simulaton loop... it works as long as the error is greater
% than 10^-6 and the number of mpc steps is less than its maximum
% value.
main_loop = tic;
while(norm((x0-xs),2) > 1e-2 && mpciter < sim_tim / h)
    % update previous control
    if mpciter == 0
        ukm1 = zeros(n_control,1);
    else
        ukm1 = u_cl(mpciter, :)';
    end
    % update parameter vector
    args.p   = [x0; xs; ukm1];
    % initial value of the optimization variables
    args.x0  = [reshape(X0',n_state*(Np+1),1);reshape(u0',n_control*Nc,1); zeros(n_control*Nc,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    % get controls only from the solution
    du = reshape(full(sol.x(n_state*(Np+1)+1:n_state*(Np+1)+n_control*Nc))',n_control,Nc)';
    % get solution TRAJECTORY
    xx1(:,1:n_state,mpciter+1) = reshape(full(sol.x(1:n_state*(Np+1)))',n_state,Np+1)';
    % store optimal control
    u_cl= [u_cl ; ukm1'+du(1,:)];
    % store slack variables
    sc_vec = reshape(full(sol.x(end-n_control*Nc+1:end))',n_control,Nc)';
    sc_cl = [sc_cl; sc_vec(1,:)];
    % time logging
    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift_delta(h, t0, x0, ukm1, du, f);
    xx(:,mpciter+2) = x0;
    X0 = reshape(full(sol.x(1:n_state*(Np+1)))',n_state,Np+1)'; % get solution TRAJECTORY
    % Shift trajectory to initialize the next step
    X0 = [X0(2:end,:); X0(end,:)];
    mpciter
    mpciter = mpciter + 1;
end
main_loop_time = toc(main_loop);

ss_error = norm((x0-xs),2)

average_mpc_time = main_loop_time/(mpciter+1)

Draw_MPC_PS_Obstacles (t,xx,xx1,u_cl,xs,Np,rob_diam,obs_x,obs_y,obs_diam)

%% Plots
Uvec = u_cl';
delta_Uvec = diff(Uvec,1,2);

figure; 
subplot(211); hold on; 
p11 = stairs(t, Uvec(1,:),'-m','LineWidth',1);
p12 = stairs(t, [Uvec(1,1), delta_Uvec(1,:)],'-b','LineWidth',1); 
p13 = stairs(t, v_max*ones(size(t)),'LineStyle','--','LineWidth',1,'Color','red'); 
stairs(t, v_min*ones(size(t)),'LineStyle','--','LineWidth',1,'Color','red');
p14 = stairs(t,dv_max*ones(size(t)),'--k','LineWidth',1); 
stairs(t,dv_min*ones(size(t)),'--k','LineWidth',1);
p15 = stairs(t,(dv_max+Sc_max(1,1))*ones(size(t)),'LineStyle','-.','LineWidth',0.5,'Color','green'); 
stairs(t,(dv_min+Sc_min(1,1))*ones(size(t)),'LineStyle','-.','LineWidth',0.5,'Color','green');
legend([p11, p12, p13, p14, p15],{'$v$','$\delta v$','$| v |$','$| \delta v |$','$| \delta v | + \mathrm{slack}$'},'interpreter','latex');

subplot(212); hold on; 
p21 = stairs(t, Uvec(2,:),'-m','LineWidth',1);
p22 = stairs(t, [Uvec(2,1),delta_Uvec(2,:)],'-b','LineWidth',1);
p23 = stairs(t, omega_max*ones(size(t)),'LineStyle','--','LineWidth',1,'Color','red'); 
stairs(t, omega_min*ones(size(t)),'LineStyle','--','LineWidth',1,'Color','red');
p24 = stairs(t,domega_max*ones(size(t)),'--k','LineWidth',1); 
stairs(t,domega_min*ones(size(t)),'--k','LineWidth',1);
p25 = stairs(t,(domega_max+Sc_max(2,1))*ones(size(t)),'LineStyle','-.','LineWidth',0.5,'Color','green'); 
stairs(t,(domega_min+Sc_min(2,1))*ones(size(t)),'LineStyle','-.','LineWidth',0.5,'Color','green');
legend([p21, p22, p23, p24, p25],{'$\omega$','$\delta \omega$','$| \omega |$','$| \delta \omega |$','$| \delta \omega | + \mathrm{slack}$'},'interpreter','latex');
xlabel('Time [s]','interpreter','latex')

% % Slacks
figure; 
subplot(211); stairs(t, sc_cl(:,1),'--m','LineWidth',1); hold on; 
stairs(t, Sc_max(1)*ones(size(t)),'-.k','LineWidth',1); 
stairs(t, Sc_min(1)*ones(size(t)),'-.k','LineWidth',1);
title('Slack on $v$','interpreter','latex')

subplot(212); stairs(t, sc_cl(:,2),'--m','LineWidth',1); hold on; 
stairs(t, Sc_max(2)*ones(size(t)),'-.k','LineWidth',1);
stairs(t, Sc_min(2)*ones(size(t)),'-.k','LineWidth',1);
xlabel('Time [s]','interpreter','latex')
title('Slack on $\omega$','interpreter','latex')