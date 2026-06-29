function [fMin, bestX, Convergence_curve] = NOA(pop_size, iter_max, lb, ub, dim, fobj, func_num)
% 文献来源：Knowledge-Based Systems 262 (2023) 110248
% 核心参数设置（严格匹配文献Table 2及3.3节默认值）
Pa1 = 0.2;         % 觅食-存储策略中探索/开发切换概率（文献5.1节最优值）
Pa2 = 0.4;         % 缓存搜索-恢复策略中探索/开发切换概率（文献5.1节最优值）
delta = 0.05;      % 全局搜索边界调节参数（文献5.1节最优值）
tau_range = [0, 1]; % 随机参数tau1-tau6范围
r_range = [0, 1];  % 随机参数r1-r3范围
alpha_range = [0, 1]; % 参考点更新系数alpha范围
Convergence_curve = zeros(1, iter_max);

% 初始化边界（支持标量/向量输入）
if isscalar(lb), lb = lb * ones(1, dim); elseif iscolumn(lb), lb = lb(:)'; end
if isscalar(ub), ub = ub * ones(1, dim); elseif iscolumn(ub), ub = ub(:)'; end

% 关键修复：CEC17函数仅支持维度D=2,10,30,50,100，添加维度校验与适配
%if func_num ~= 0 && ~ismember(dim, [2,10,30,50,100])
%    error('CEC17函数仅支持维度D=2,10,30,50,100，请调整dim参数后重新运行');
%end

% 种群初始化（文献4节公式19）
x = lb + (ub - lb) .* rand(pop_size, dim);

% 初始适应度计算（适配CEC17函数维度限制与工程优化）
if func_num == 0  % 工程优化（直接计算）
    fit = feval(fobj, x');
else              % CEC17（仅支持指定维度，传入函数编号）
    fit = feval(fobj, x', func_num);
end

% 个体最优与全局最优初始化
pFit = fit;
pX = x;
[fMin, bestIdx] = min(fit);
bestX = x(bestIdx, :);

% 参考点（RPs）初始化（每个个体2个参考点，文献3.3.3节）
RPs = zeros(pop_size, 2, dim);
for i = 1:pop_size
    RPs(i, 1, :) = lb + (ub - lb) .* rand(1, dim);
    RPs(i, 2, :) = lb + (ub - lb) .* rand(1, dim);
end

for t = 1:iter_max
    % 1. 参数动态更新（文献3.3.3节）
    tau1 = tau_range(1) + (tau_range(2)-tau_range(1)) * rand();
    tau2 = tau_range(1) + (tau_range(2)-tau_range(1)) * rand();
    tau3 = tau_range(1) + (tau_range(2)-tau_range(1)) * rand();
    r1 = r_range(1) + (r_range(2)-r_range(1)) * rand();
    r2 = r_range(1) + (r_range(2)-r_range(1)) * rand();
    r3 = r_range(1) + (r_range(2)-r_range(1)) * rand();
    theta = rand() * pi;  % 视角角度（0~pi，文献3.3.3节）
    
    % alpha动态更新（文献3.3.3节公式11）
    if r1 > r2
        alpha = (1 - t/iter_max)^(2*t/iter_max);  % 递减策略
    else
        alpha = (t/iter_max)^(2/t);  % 递增策略（避免局部最优）
    end
    alpha = max(alpha, 0.01);  % 防止alpha过小
    
    % 2. 随机选择当前策略（觅食-存储 / 缓存搜索-恢复，概率各50%）
    strategy = randi([1, 2]);
    
    if strategy == 1  % 策略1：觅食-存储策略（文献3.3.1节）
        for i = 1:pop_size
            % 随机选择3个不同个体（A/B/C）
            A = randi(pop_size);
            while A == i, A = randi(pop_size); end
            B = randi(pop_size);
            while B == i || B == A, B = randi(pop_size); end
            C = randi(pop_size);
            while C == i || C == A || C == B, C = randi(pop_size); end
            
            % 计算mu参数（文献3.3.1节公式2）
            if r1 < r2
                mu = rand();  % tau3随机值
            elseif r2 < r3
                mu = randn(); % tau4正态分布
            else
                mu = levy_flight(1, 1.5);  % tau5 Lévy飞行
            end
            mu = abs(mu);  % 取绝对值保证步长有效性
            
            % 探索阶段1（觅食行为，文献3.3.1节公式1）
            if rand() > Pa1
                if tau1 < tau2
                    x_new = x(i, :);  % 发现优质种子，保持位置
                else
                    if t <= iter_max/2
                        % 前期：全局探索（基于3个个体）
                        gamma = levy_flight(1, 1.5);  % Lévy飞行系数
                        x_new = pX(A, :) + gamma .* (pX(B, :) - pX(C, :)) + mu .* (r3^2 .* (ub - lb));
                    else
                        % 后期：局部探索（增加边界约束）
                        x_new = pX(C, :) + mu .* (pX(A, :) - pX(B, :)) + mu .* (rand() < delta) .* (r3^2 .* (ub - lb));
                    end
                end
            else  % 开发阶段1（存储行为，文献3.3.1节公式3）
                lambda = levy_flight(1, 1.5);  % Lévy飞行系数
                if tau1 < tau2
                    x_new = x(i, :) + mu .* (bestX - x(i, :)) .* abs(lambda) + r1 .* (pX(A, :) - pX(B, :));
                elseif tau1 < tau3
                    x_new = bestX + mu .* (pX(A, :) - pX(B, :));
                else
                    l = 1 - t/iter_max;  % 线性递减因子
                    x_new = bestX .* l;
                end
            end
            
            % 边界控制
            x_new = Bounds_NOA(x_new, lb, ub);
            x(i, :) = x_new;
        end
    else  % 策略2：缓存搜索-恢复策略（文献3.3.3节）
        % 更新参考点RPs（文献3.3.3节公式9-10）
        for i = 1:pop_size
            % 随机选择2个不同个体
            A = randi(pop_size);
            while A == i, A = randi(pop_size); end
            B = randi(pop_size);
            while B == i || B == A, B = randi(pop_size); end
            
            % 第一个参考点更新
            if theta == pi/2
                RP1 = x(i, :) + alpha .* cos(theta) .* (pX(A, :) - pX(B, :)) + alpha .* (lb + (ub - lb).*rand(1, dim));
            else
                RP1 = x(i, :) + alpha .* cos(theta) .* (pX(A, :) - pX(B, :));
            end
            RP1 = Bounds_NOA(RP1, lb, ub);
            RPs(i, 1, : ) = RP1;
            
            % 第二个参考点更新
            U2 = rand(1, dim) < 0.5;
            if theta == pi/2
                RP2 = x(i, : ) + alpha .* cos(theta) .* ((ub - lb).*rand(1, dim) + lb) .* U2 + alpha .* (lb + (ub - lb).*rand(1, dim));
            else
                RP2 = x(i, :) + alpha .* cos(theta) .* ((ub - lb).*rand(1, dim) + lb) .* U2;
            end
            RP2 = Bounds_NOA(RP2, lb, ub);
            RPs(i, 2, :) = RP2;
        end
        
        % 种群位置更新
        for i = 1:pop_size
            % 探索阶段2（缓存搜索，文献3.3.3节公式12/14）
            if rand() > Pa2
                % 提取参考点并转置为行向量（匹配x(i,:)格式）
                rp1_vec = squeeze(RPs(i, 1, :))';  % ← 添加转置，变成1×dim
                rp2_vec = squeeze(RPs(i, 2, :))';  % ← 添加转置，变成1×dim
                
                % 计算适应度（CEC17需要列向量，所以再转置回去）
                if func_num == 0
                    fit_RP1 = feval(fobj, rp1_vec');
                    fit_RP2 = feval(fobj, rp2_vec');
                else
                    fit_RP1 = feval(fobj, rp1_vec', func_num);
                    fit_RP2 = feval(fobj, rp2_vec', func_num);
                end
                
                % 选择更优参考点更新
                if fit_RP1 < pFit(i)
                    x_new = rp1_vec;  % 已经是1×dim行向量
                elseif fit_RP2 < pFit(i)
                    x_new = rp2_vec;  % 已经是1×dim行向量
                else
                    x_new = lb + (ub - lb) .* rand(1, dim);
                end
            else  % 开发阶段2（缓存恢复，文献3.3.3节公式13/15）
                C = randi(pop_size);
                while C == i, C = randi(pop_size); end
                tau5 = tau_range(1) + (tau_range(2)-tau_range(1)) * rand();
                tau6 = tau_range(1) + (tau_range(2)-tau_range(1)) * rand();
                
                if tau5 < tau6
                    % 修复：将RPs提取后转置为行向量
                    rp2_vec = squeeze(RPs(i, 2, :))';  % ← 添加转置
                    x_new = x(i, :) + r1 .* (bestX - x(i, :)) + r2 .* (rp2_vec - pX(C, : ));
                else
                    x_new = lb + (ub - lb) .* rand(1, dim);
                end
            end
            
            % 边界控制
            x_new = Bounds_NOA(x_new, lb, ub);
            x(i, :) = x_new;
        end
    end
    
    % 3. 适应度更新
    if func_num == 0
        fit_new = feval(fobj, x');
    else
        fit_new = feval(fobj, x', func_num);
    end
    
    % 4. 个体最优与全局最优更新（文献4节公式20）
    for i = 1:pop_size
        if fit_new(i) < pFit(i)
            pFit(i) = fit_new(i);
            pX(i, :) = x(i, :);
        end
        if pFit(i) < fMin
            fMin = pFit(i);
            bestX = pX(i, :);
        end
    end
    fit = fit_new;
    
    % 5. 记录收敛曲线
    Convergence_curve(t) = fMin;
end
end

% 辅助函数1：边界控制（文献默认策略）
function s = Bounds_NOA(s, Lb, Ub)
    s = max(s, Lb);
    s = min(s, Ub);
end

% 辅助函数2：Lévy飞行生成（文献3.3节公式）
function levy = levy_flight(dim, theta)
    sigma = (gamma(1+theta)*sin(pi*theta/2)/(gamma((1+theta)/2)*theta*2^((theta-1)/2)))^(1/theta);
    levy = 0.05 * randn(1, dim) * sigma ./ (abs(randn(1, dim)).^(1/theta));
end
