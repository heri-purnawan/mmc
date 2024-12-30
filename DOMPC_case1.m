clc; clear all;
%define the matrices A and B in continuous time
Ac = [-0.1068 0 ; 1 0];
BGc = [0.0028 1; 0 0];
C = [1 0; 0 1];  %output matrix

dt = 0.5; %sampling time
%obtain matrices A and B in discrete time
[A, BG] = c2d(Ac,BGc,dt);
B = BG(:,1); G = BG(:,2);
%prediction horizon
N = 40;
%define initial conditions for state variables
x0 = [0;30];
%define input constraints
rudder_max = 35; rudder_min = -rudder_max;
r_max = 0.34;  r_min = -r_max;

E = [1 0;-1 0]; F = [r_max;-r_min] ;
S = [1; -1]; T = [rudder_max;-rudder_min];

%********** define matrix Qbar and Rbar *************
q = [1000,300]; r = 1;
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
h1(1:2,:) = E;
gamma(1:2,:) = E*A;
for i = 2:N
    h1(2*i-1:2*i,:) = h1(2*i-3:2*i-2,:)*A;
    gamma(2*i-1:2*i,:) = gamma(2*i-3:2*i-2,:)*A;
end

%define matrix phi
phi = zeros(2*N,N);
L1 = h1*B;
phi(:,1) = L1;
for i = 2:N
    phi(:,i) = [zeros(2*i-2,1);L1(1:2*(N-i+1),:)];
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

%% =========================== Scheme 1 ===============================
%=========================== define Psi_1 ====================
Psi_1(1:2,:) = C;
I = eye(2);
lamb_1 = eye(2);
for i = 2:N
    I = I*A;
    lamb_1 = I + lamb_1;
    Psi_1(2*i-1:2*i,:) = C*lamb_1; 
end
%==============================================================

%=============== define psi_1 =============================
psi_1(1:2,:) = E;
I = eye(2);
lamb_2 = eye(2);
for i = 2:N
    I = I*A;
    lamb_2 = I + lamb_2;
    psi_1(2*i-1:2*i,:) = E*lamb_2; 
end
%===================================================================

%% =========================== Scheme 2 ===============================
%%**************** define matrix Psi_2 *******************
% Psi_2(1:2,:) = C;
% for i = 2:N
%     Psi_2(i*2-1:i*2,:) = Psi_2(i*2-3:i*2-2,:)*A;
% end
% %%==============================================================
% 
% %%=================== define psi_2 =============================
% psi_2(1:2,:) = E;
% for i = 2:N
%     psi_2(i*2-1:i*2,:) = psi_2(i*2-3:i*2-2,:)*A;
% end
%%=======================================================================

%******************* DO *******************
dhat(1) = 0;
% z(1) = 0;
%determine matrix M
M = eye(2);
Lambda = 0.75;
%calculate K
I_w = pinv(M*G)*(M*G);
K = (I_w - Lambda)*pinv(M*G)*M;
z(1) = K*x0 - dhat(1);

tf = 300;
N_iter = tf/dt;
time = linspace(0,tf,N_iter);
k = 1;
x(:,1) = x0;
u0 = zeros(N,1);
d(k) = 0.01*sin(0.08*k*dt);

while k <= N_iter

    yref = zeros(2*N,1) ; %referensi

    psi_ref(k) = 0; %only for plotting

    %update matrix F_1
    F_1 = 2*Phi'*Qbar*(Omega*x(:,k) + Psi_1*G*dhat(k) - yref);
    % F_2 = 2*Phi'*Qbar*(Omega*x(:,k) + Psi_2*G*dhat(k) - yref);

    %%optimization stage
    %% ===== Scheme 1 =====
    u_star = quadprog(H,F_1,[phi; Sbar],[Fbar-gamma*x(:,k)-psi_1*G*dhat(k);Tbar],[],[],[],[],u0);
    %% ===== Scheme 2 =====
    % u_star = quadprog(H,F_2,[phi; Sbar],[Fbar-gamma*x(:,k)-psi_2*G*dhat(k);Tbar],[],[],[],[],u0);

    u0 = u_star;
    u(1,k) = u_star(1);

    x(:,k+1) = A*x(:,k) + B*u(1,k) + G*d(k);

    %update k
    k = k + 1;
    d(k) = 0.01*sin(0.08*k*dt);

    %calculate disturbance estimate 
    z(k) = z(k-1) + K*((A-eye(2))*x(:,k-1) + B*u(1,k-1)+ G*dhat(k-1));
    dhat(k) = K*x(:,k) - z(k);

end
Draw_DOMPC_case1(time,x,u,psi_ref,d,dhat)