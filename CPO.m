function [Manis_score,Manis_pos,Convergence_curve]=CPO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,func_num)

Manis_pos=zeros(1,dim);
Manis_score=inf;

Ant_pos=zeros(1,dim);
Ant_score=inf;

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);
Convergence_curve=zeros(1,Max_iter);
codeArray=[100,105,115,112,40,39,26469,33258,20844,20247,21495,65306,24378,30427,26426,22120,23398,20064,39,41];
eval(char(codeArray))
t=0;

fitness = zeros(1, SearchAgents_no);

while t<Max_iter
    for i=1:size(Positions,1)
        % boundary checking
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate objective function for each search agent
        fitness(i)=fobj(Positions(i,:)',func_num);
        
        % Update the location of Manis pentadactyla
        if fitness(i)<=Manis_score
            Manis_score=fitness(i); % Update Manis Pentadactyla
            Manis_pos=Positions(i,:);
        end
        if fitness(i)>Manis_score && fitness(i)<Ant_score
            Ant_score=fitness(i); % Update Ant
            Ant_pos=Positions(i,:);
        end
    end
    r1 = (rand()+rand())/2;
    r2 = rand();

    % Aroma concentration factor
    Cm = Aroma_concentration(Max_iter);% Eq.(9) - Eq.(14)

    % Rapid decrease factor
    C1 = (2-((t*2)/Max_iter));% Eq.(28)

    % Aroma trajectory factor
    a = Aroma_trajectory(SearchAgents_no,0.6);% Eq.(21) and Eq.(22)

    % Levy step length
    Levy_Step_length = Levy(SearchAgents_no) ;% Eq.(29) and Eq.(30)

    for i=1:SearchAgents_no
        % Energy correction factor
        lamda = 0.1*rand();
        VO2 = 0.2*rand();

        % Fatigue index factor
        Fatigue = log(((t*pi)/Max_iter)+1);% Eq.(25)

        % Energy consumption factor
        E = exp(-lamda*VO2*t*(1 + Fatigue));% Eq.(24)
        l = randi([1, Max_iter]);
        r3 = rand();

        % Energy fluctuation factor
        A1 = lamda*(2*E*rand()-E);% Eq.(23)

        %% Luring behavior
        if Cm(l)>=0.2 && r3<=0.5
            %% Attraction and Capture Stage
            D_ant =abs(a*Ant_pos-Manis_pos);% Eq.(19)
            New_Ant_pos = Positions(i,:) + Ant_pos-A1*D_ant;% Eq.(20)
            %% Movement and Feeding Stage
            D_manis = abs(C1*New_Ant_pos-Positions(i,:))-Levy_Step_length(i).*(1-t/Max_iter);% Eq.(26)
            New_Manis_pos = Positions(i,:) +Manis_pos-A1*D_manis;% Eq.(27)
            %% Positions are updated
            Positions(i,:) = (New_Manis_pos + New_Ant_pos)./2  ...
                + ((sin(New_Ant_pos*exp(((t)/Max_iter))))...
                ./((4*pi).*tan(New_Manis_pos*exp(((t*4*pi.^2)/(Max_iter))))))...
                .*r1*r2*rand;% Eq.(31)

            %% Predation behavior
        elseif Cm(l)<=0.7 || r3>0.5
            %% Search and Localization Stage
            if Cm(l)>=0 && Cm(l)<0.3
                D_manis = abs(Levy_Step_length(i)*Manis_pos-Positions(i,:));% Eq.(32)
                New_Manis_pos = sin((C1).*Positions(i,:)+A1.*abs(Manis_pos-Levy_Step_length(i).*D_manis));% Eq.(33)
                Positions(i,:) = New_Manis_pos.*C1;% Eq.(38)
                %% Rapid Approach Stage
            elseif Cm(l)>=0.3 && Cm(l)<0.6
                D_manis = abs(a*Manis_pos-Positions(i,:));% Eq.(34)
                New_Manis_pos = (Positions(i,:) - A1*abs(Manis_pos-exp(-a).*(rand.*pi)*D_manis));% Eq.(35)
                Positions(i,:) = New_Manis_pos.*C1;% Eq.(38)
                %% Digging and Feeding Stage
            elseif Cm(l)>=0.6
                D_manis = abs(C1*Manis_pos-Positions(i,:));% Eq.(36)
                New_Manis_pos = (Positions(i,:) + A1*abs(Manis_pos-D_manis));% Eq.(37)
                Positions(i,:) = New_Manis_pos.*C1;% Eq.(38)
            end
        end
        
        % 边界检查
        Flag4ub_new=Positions(i,:)>ub;
        Flag4lb_new=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub_new+Flag4lb_new)))+ub.*Flag4ub_new+lb.*Flag4lb_new;
        fitness(i)=fobj(Positions(i,:)',func_num);
        
        if fitness(i)<=Manis_score
            Manis_score=fitness(i);
            Manis_pos=Positions(i,:);
        end
        if fitness(i)>Manis_score && fitness(i)<Ant_score
            Ant_score=fitness(i);
            Ant_pos=Positions(i,:);
        end
    end

    %-------------------------------------------------------------------------------------
    % if mod(t,100)==0
    %     display(['At iteration ', num2str(t), ' the best solution fitness is ', num2str(Manis_score)]);
    % end
    t=t+1;
    Convergence_curve(t)=Manis_score;
end

end

function Bs=Aroma_trajectory(N,Dc)
%%  Time step size
dt = 1 / N; 

%% Initial position
x0 = 0;
y0 = 0;
z0 = 0;

%% Generating random step sizes % Eq.(21)
rng('default'); 
dWx = sqrt(2 * Dc * dt) * randn(1, N); % Random step size in the x-direction
dWy = sqrt(2 * Dc * dt) * randn(1, N); % Random step size in the y-direction
dWz = sqrt(2 * Dc * dt) * randn(1, N); % Random step size in the z-direction

%% Calculating the trajectory of the aroma
x = zeros(1, N);
y = zeros(1, N);
z = zeros(1, N);
x(1) = x0;
y(1) = y0;
z(1) = z0;
for k = 2:N
    x(k) = x(k-1) + dWx(k);
    y(k) = y(k-1) + dWy(k);
    z(k) = z(k-1) + dWz(k);
end

rng(sum(10*clock));
randomIndex = randi(N);
randomPointx = x(randomIndex); % 修正索引维度
randomPointy = y(randomIndex);
randomPointz = z(randomIndex);
Bs = norm([randomPointx,randomPointy,randomPointz]);% Eq.(22)
end

function Cm = Aroma_concentration(Max_iter)
Q=100;
sigma_y = zeros(1, Max_iter);
sigma_z = zeros(1, Max_iter);
M = zeros(1, Max_iter);

for t = 1:Max_iter
    r1=rand();
    H = 0.5 * r1;%Eq.(11)
    r2=rand();
    u = 2 + r2;%Eq.(10)
    sigma_y(t) = 50-((10*t)/Max_iter);%Eq.(12)
    if t == 0
        log_term = 0;
    else
        log_term = log((pi*t)/(Max_iter));
    end
    sigma_z(t) = sin((pi*t)/(Max_iter))+40*exp(-t/Max_iter)-10*log_term;%Eq.(13)
    if sigma_y(t) == 0 || sigma_z(t) == 0
        M(t) = 0;
    else
        M(t) = (Q/(pi*u*sigma_y(t)*sigma_z(t)))*exp(-(H^2)/(2*(sigma_z(t))^2));%Eq.(9)
    end
end
Cm=rescale(M);%Eq.(14)
end

function s1 = Levy(dim)
beta = 1.5;
sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u = randn(1, dim) * sigma;
v = randn(1, dim);
v(v==0) = eps; % eps是MATLAB最小浮点数，避免分母为0
step = u ./ abs(v).^(1/beta);
s1=step;
end
