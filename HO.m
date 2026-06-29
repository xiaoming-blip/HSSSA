function [fMin, bestX, Convergence_curve] = HO(pop_size, iter_max, lb, ub, dim, fobj, func_num)
% 文献来源：Scientific Reports (2024) 14:5032
% 核心参数设置（严格匹配文献2.3节及Table 1默认值）
theta = 1.5;        % Lévy飞行参数（文献2.3.2节）
rho_range = [2, 4]; % 防御阶段参数ρ范围
psi_range = [1, 1.5];% 防御阶段参数ψ范围
delta_range = [2, 3];% 防御阶段参数δ范围
Convergence_curve = zeros(1, iter_max);

bestX = zeros(1, dim);

% 初始化边界与种群
if isscalar(lb), lb = lb * ones(1, dim); elseif iscolumn(lb), lb = lb(:)'; end
if isscalar(ub), ub = ub * ones(1, dim); elseif iscolumn(ub), ub = ub(:)'; end
x = lb + (ub - lb) .* rand(pop_size, dim);  % 种群位置初始化（文献2.3.1节公式1）

% 初始适应度计算（适配CEC17函数编号）
if func_num == 0  % 工程优化（直接计算）
    fit = feval(fobj, x');
else              % CEC17（需传入函数编号）
    fit = feval(fobj, x', func_num);
end

% 初始化个体最优与全局最优（主导河马=全局最优）
pFit = fit;
pX = x;
[fMin, dom_idx] = min(fit);
domX = x(dom_idx, :);  % 主导河马位置（文献2.3.1节）
bestX=domX;

for t = 1:iter_max
    % 1. 阶段1：河马在水域中的位置更新（探索，文献2.3.1节）
    T = exp(-t / iter_max);  % 文献2.3.1节公式5（动态参数）
    
    % 雄性河马更新（前50%种群）
    male_idx = 1:round(pop_size/2);
    for i = male_idx
        % 随机参数生成（文献2.3.1节公式4）
        r1 = rand(1, dim); r2 = rand(1, dim); r3 = rand(1, dim); r4 = rand(1, dim); r5 = rand();
        rho1 = randi([0,1], 1, dim); rho2 = randi([0,1], 1, dim);
        h = zeros(1, dim);
        for d = 1:dim
            h(d) = select_h(r1(d), r2(d), r3(d), r4(d), r5, rho1(d), rho2(d));
        end
        
        % 随机选择参考河马（文献2.3.1节公式3）
        rand_idx = randi(pop_size, 1, 3);
        mean_X = mean(pX(rand_idx, :), 1);
        x(i, :) = domX + h .* (mean_X - randi([1,2]) * x(i, :));
        
        % 边界控制
        x(i, :) = Bounds_HO(x(i, :), lb, ub);
    end
    
    % 雌性/幼年河马更新（后50%种群）
    female_idx = round(pop_size/2)+1:pop_size;
    for i = female_idx
        r6 = rand();
        if T > 0.6 || r6 > 0.5
            % 远离母亲/群体（文献2.3.1节公式6-7）
            w = rand(1, dim) * 2;  % w∈[0,2]
            x(i, :) = domX + w .* (x(i, :) - lb) + (1 - w) .* (x(i, :) - ub);
        else
            % 靠近母亲/群体
            x(i, :) = domX + rand(1, dim) .* (pX(i, :) - domX);
        end
        x(i, :) = Bounds_HO(x(i, :), lb, ub);
    end
    
    % 2. 阶段2：防御行为（探索，文献2.3.2节）
    predator = lb + rand(1, dim) .* (ub - lb);  % 捕食者位置（公式10）
    % 捕食者适应度计算
    if func_num == 0
        predator_fit = feval(fobj, predator');
    else
        predator_fit = feval(fobj, predator', func_num);
    end
    
    for i = 1:pop_size
        D = abs(predator - x(i, :));  % 与捕食者距离（公式11）
        RL = levy_flight(dim, theta); % Lévy飞行扰动（修正后函数）
        rho = rho_range(1) + (rho_range(2)-rho_range(1))*rand();
        psi = psi_range(1) + (psi_range(2)-psi_range(1))*rand();
        
        % 防御位置更新（公式12）
        if predator_fit >= pFit(i)
            x(i, :) = RL .* predator + (t / (-iter_max * cos(2*pi*rand()))) ./ (2*D + rand(1, dim));
        else
            x(i, :) = RL .* predator;
        end
        x(i, :) = Bounds_HO(x(i, :), lb, ub);
    end
    
    % 3. 阶段3：逃离行为（开发，文献2.3.3节）
    H_ind = rand(pop_size, dim) / t;  % 个体动态边界（公式16）
    H_cont = rand(1, dim) / t;   % 群体动态边界
    for i = 1:pop_size
        % 选择逃离策略（公式18）
        s_type = randi(3);
        switch s_type
            case 1
                s = 2 * rand(1, dim) - 1;  % 策略1：[-1,1]随机
            case 2
                s = rand();  % 策略2：[0,1]随机
            case 3
                s = randn(1, dim);  % 策略3：正态分布
        end
        % 逃离位置更新（公式17）
        x(i, :) = x(i, :) + s .* (H_ind(i, :) + H_cont - x(i, :));
        x(i, :) = Bounds_HO(x(i, :), lb, ub);
    end
    
    % 4. 适应度与最优解更新
    if func_num == 0
        fit_new = feval(fobj, x');
    else
        fit_new = feval(fobj, x', func_num);
    end
    
    for i = 1:pop_size
        if fit_new(i) < pFit(i)
            pFit(i) = fit_new(i);
            pX(i, :) = x(i, :);
        end
        if pFit(i) < fMin
            fMin = pFit(i);
            bestX = pX(i, :);
            domX = bestX;  % 更新主导河马位置
        end
    end
    fit = fit_new;
    
    % 5. 记录收敛曲线
    Convergence_curve(t) = fMin;
end
end

% 辅助函数1：HO边界控制（文献2.3节默认策略）
function s = Bounds_HO(s, Lb, Ub)
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);
    J = temp > Ub;
    temp(J) = Ub(J);
    s = temp;
end

% 辅助函数2：Lévy飞行生成（核心修正：^→.^）（文献2.3.2节公式13-14）
function levy = levy_flight(dim, theta)
    % 修正点：将 abs(randn(1, dim))^(1/theta) 改为 abs(randn(1, dim)).^(1/theta)
    sigma = (gamma(1+theta)*sin(pi*theta/2)/(gamma((1+theta)/2)*theta*2^((theta-1)/2)))^(1/theta);
    levy = 0.05 * randn(1, dim) * sigma ./ (abs(randn(1, dim)).^(1/theta));
end

% 辅助函数3：h参数选择（文献2.3.1节公式4）
function h_val = select_h(r1, r2, r3, r4, r5, rho1, rho2)
    if r5 <= 0.2
        h_val = randi([1,2]) * r1 + (1 - rho1);
    elseif r5 <= 0.4
        h_val = 2 * r2 - 1;
    elseif r5 <= 0.6
        h_val = r3;
    elseif r5 <= 0.8
        h_val = randi([1,2]) * r4 + (1 - rho2);
    else
        h_val = r5;
    end
end
