clc; clear all;
%define the matrices A and B in continuous time
Ac = [-5 -0.3427;
        47.68 2.785];
BGc = [0 1; 0.3 0];
% B = BGc(:,1); G = BGc(:,2);
C = [1 0;0 1];  %output matrix

dt = 0.5; %sampling time
[A,BG] = c2d(Ac,BGc,dt);
B = BG(:,1); G = BG(:,2);
%prediction horizon
N = 15;
%define initial conditions for state variables
x0 = [0;0];
%define input and state constraints
Tc_max = 10; Tc_min = -10;
CA_min = -0.2;

E = [-1 0]; F = -CA_min;
S = [1; -1]; T = [Tc_max;-Tc_min];
%********** define matrix Qbar and Rbar *************
q = [0,1]; r = 0;
Q = diag(q); R = diag(r);
Qbar = zeros(2*N,2*N); Rbar = zeros(N,N);
for i = 1:N
    Qbar(i*2-1:i*2,i*2-1:i*2) = Q;
    Rbar(i,i) = R;
end
%====================================================

%**************** define matrix Omega *******************
h(1:2,:) = C;
Omega(1:2,:) = C*A;
for i = 2:N
    h(i*2-1:i*2,:) = h(i*2-3:i*2-2,:)*A;
    Omega(i*2-1:i*2,:) = Omega(i*2-3:i*2-2,:)*A;
end
%=====================================================

%**************** define matrix Phi *****************
L = h*B;
Phi = zeros(2*N,N);
Phi(:,1) = L;
for i = 2:N
    Phi(:,i) = [zeros(2*i-2,1);L(1:2*(N-i+1),:)];
end
%=====================================================

%define matrix gamma
h1(1,:) = E;
gamma(1,:) = E*A;
for i = 2:N
    h1(i,:) = h1(i-1,:)*A;
    gamma(i,:) = gamma(i-1,:)*A;
end

%define matrix phi
phi = zeros(N,N);
L1 = h1*B;
phi(:,1) = L1;
for i = 2:N
    phi(:,i) = [zeros(i,1);L1(1:(N-i),:)];
end

%define matrix Fbar
Fbar = [];
for i = 1:N
    Fbar = [Fbar;F];
end

%==================================================================
%***** define lower and upper bound for input ******
Sbar = zeros(2*N,N);
for i = 1:N
    Sbar(2*i-1:2*i,i) = S; 
end

%define matrix Tbar
Tbar = [];
for i = 1:N
    Tbar = [Tbar;T];
end
%====================================================================

%***************** define matrix H *******************
H = 2*(Phi'*Qbar*Phi+Rbar);
%% ============================= Scheme 1 ============================
% %=========================== define Psi_1 ====================
% Psi_1(1:2,:) = C;
% I = eye(2);
% lamb_1 = eye(2);
% for i = 2:N
%     I = I*A;
%     lamb_1 = I + lamb_1;
%     Psi_1(2*i-1:2*i,:) = C*lamb_1; 
% end
% %==============================================================
% 
% %=============== define psi_1 =============================
% psi_1(1,:) = E;
% I = 1;
% lamb_2 = 1;
% for i = 2:N
%     I = I*A;
%     lamb_2 = I + lamb_2;
%     psi_1(i,:) = E*lamb_2; 
% end
% %====================================================================
%% ============================= Scheme 2 ============================
%**************** define matrix Psi_2 *******************
Psi_2(1:2,:) = C;
for i = 2:N
    Psi_2(i*2-1:i*2,:) = Psi_2(i*2-3:i*2-2,:)*A;
end
%==============================================================

%=================== define psi_2 =============================
psi_2(1,:) = E;
for i = 2:N
    psi_2(i,:) = psi_2(i-1,:)*A;
end
%=======================================================================

%******************* DO *******************
dhat(1) = 0;
% z(1) = 0;
%determine matrix M
M = eye(2);
Lambda = 0.25;
%calculate K
I_w = pinv(M*G)*(M*G);
K = (I_w - Lambda)*pinv(M*G)*M;
z(1) = K*x0 - dhat(1);

tf = 20;
N_iter = tf/dt;
time = linspace(0,tf,N_iter);
k = 1;
x(:,1) = x0;
u0 = zeros(N,1);
if time(k) <= 5
    d(k) = 0;
else
    d(k) = 0.2;
end

% Referensi setpoint (diskrit)
r = @(t) (t > 13) + (t >= 5 & t <= 13) * 2; % step with 3 levels

% Initialization for all iterations
yref = zeros(2 * N, 1); % array referensi

while k <= N_iter

    %reference
    for i = 1:N
        t_pred = (k - 1) * dt + (i - 1) * dt; % prediction time
        yref(2 * i - 1:2 * i) = [0;r(t_pred)]; % [0;ref]
    end
    
    %only for plotting
    if time(k) <= 5
        T_ref(k) = 0;   % T_ref = 0 for t <= 5
    elseif time(k) <= 13
        T_ref(k) = 2;   % T_ref = 2 for t > 5 & t <= 12
    else
        T_ref(k) = 1;
    end

    %update matrix F_1
    % F_1 = 2*Phi'*Qbar*(Omega*x(:,k) + Psi_1*G*dhat(k) - yref);
    F_2 = 2*Phi'*Qbar*(Omega*x(:,k) + Psi_2*G*dhat(k) - yref);

    %optimization stage
    % u_star = quadprog(H,F_1,[phi; Sbar],[Fbar-gamma*x(:,k)-psi_1*G*dhat(k);Tbar],[],[],[],[],u0);
    u_star = quadprog(H,F_2,[phi; Sbar],[Fbar-gamma*x(:,k)-psi_2*G*dhat(k);Tbar],[],[],[],[],u0);

    u0 = u_star;
    u(1,k) = u_star(1);

    x(:,k+1) = A*x(:,k) + B*u(1,k) + G*d(k);

    %update k
    if time(k) <= 5
        d(k+1) = 0;
    else
        d(k+1) = 0.2;
    end
    k = k + 1;
    

    %calculate disturbance estimate 
    z(k) = z(k-1) + K*((A-eye(2))*x(:,k-1) + B*u(1,k-1)+ G*dhat(k-1));
    dhat(k) = K*x(:,k) - z(k);

end
Draw_DOMPC_case2(time,x,u,T_ref,d,dhat)