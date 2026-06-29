function [fMin, bestX, Convergence_curve] = ICSSA(pop, M, c, d, dim, fobj, func_num)
    % ICSSA - Improved Chaos Sparrow Search Algorithm
    % 基于文献《An improved chaos sparrow search algorithm for UAV path planning》实现
    % 包含四个改进策略：
    % 1. PWLCM混沌映射初始化种群
    % 2. 非线性动态权重因子优化发现者更新
    % 3. 改进的正弦余弦算法优化跟随者更新
    % 4. 动态边界透镜成像反向学习策略
    
    % ========== 参数设置 ==========
    P_percent = 0.2;    % 发现者比例
    pNum = round(pop * P_percent);   % 发现者数量
    SD = 0.2;          % 警戒者比例
    sNum = round(pop * SD);   % 警戒者数量
    
    lb = c .* ones(1, dim);    % 下界
    ub = d .* ones(1, dim);    % 上界
    
    % ========== PWLCM混沌映射初始化种群 ==========
    % 文献中使用的分段混沌映射(PWLCM)
    x = zeros(pop, dim);
    for i = 1:pop
        for j = 1:dim
            if i == 1 && j == 1
                % 初始值
                chaos_val = rand();
            else
                % PWLCM混沌映射 
                chaos_val = PWLCM(chaos_val);
            end
            x(i, j) = lb(j) + (ub(j) - lb(j)) * chaos_val;
        end
    end
    
    % ========== 初始化适应度 ==========
    fit = zeros(1, pop);
    for i = 1:pop
        fit(i) = fobj(x(i, :)', func_num);
    end
    
    pFit = fit;        % 个体历史最佳适应度
    pX = x;            % 个体历史最佳位置
    
    [fMin, bestI] = min(fit);    % 全局最佳适应度
    bestX = x(bestI, :);         % 全局最佳位置
    
    Convergence_curve = zeros(1, M);  % 收敛曲线
    
    % ========== 主循环 ==========
    for t = 1:M
        % ===== 非线性动态权重因子  =====
        w_max = 0.9;
        w_min = 0.4;
        w = w_min + (w_max - w_min) * exp(-(2 * t / M)^2);
        
        % 排序适应度
        [~, sortIndex] = sort(pFit);
        
        % 找到最差适应度个体
        [fmax, worstIdx] = max(pFit);
        worst = pX(worstIdx, :);
        
        % ===== 发现者更新 =====
        for i = 1:pNum
            r2 = rand();
            if r2 < 0.8
                r1 = rand();
                % 应用非线性动态权重因子 
                x(sortIndex(i), :) = w * pX(sortIndex(i), :) .* exp(-(i) / (r1 * M));
            else
                x(sortIndex(i), :) = w * pX(sortIndex(i), :) + randn(1, dim);
            end
            % 边界处理
            x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub);
            fit(sortIndex(i)) = fobj(x(sortIndex(i), :)', func_num);
        end
        
        % ===== 跟随者更新 (改进的正弦余弦算法) =====
        for i = (pNum + 1):pop
            % 改进的正弦余弦算法参数
            r1 = 2 * (1 - t/M);  % 线性递减
            r2 = 2 * pi * rand(); % [0, 2π]随机值
            r3 = 2 * rand();      % [0, 2]随机值 (控制移动距离)
            r4 = rand();          % [0, 1]随机值 (选择正弦或余弦)
            
            if r4 < 0.5
                % 正弦更新
                x(sortIndex(i), :) = w * pX(sortIndex(i), :) + ...
                    r1 * sin(r2) * abs(r3 * bestX - pX(sortIndex(i), :));
            else
                % 余弦更新
                x(sortIndex(i), :) = w * pX(sortIndex(i), :) + ...
                    r1 * cos(r2) * abs(r3 * bestX - pX(sortIndex(i), :));
            end
            
            % 边界处理
            x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub);
            fit(sortIndex(i)) = fobj(x(sortIndex(i), :)', func_num);
        end
        
        % ===== 警戒者更新 (文献公式3) =====
        for i = 1:sNum
            r = randperm(pop, 1);
            if pFit(r) > fMin
                % 向安全区域移动
                x(r, :) = bestX + randn(1, dim) .* abs(pX(r, :) - bestX);
            else
                % 向种群中心移动
                x(r, :) = pX(r, :) + (2 * rand(1) - 1) .* ...
                    abs(pX(r, :) - worst) / (pFit(r) - fmax + 1e-50);
            end
            % 边界处理
            x(r, :) = Bounds(x(r, :), lb, ub);
            fit(r) = fobj(x(r, :)', func_num);
        end
        
        % ===== 透镜成像反向学习策略 =====
        % 计算动态边界 
        low_bound = min(x);
        up_bound = max(x);
        
        % 缩放因子k的线性递增策略 
        d_val = 0.2;  % 文献中d=0.2
        k = d_val + (1 - d_val) * (t / M);
        
        % 对每个个体生成反向解
        for i = 1:pop
            % 透镜成像反向学习公式
            opposite_x = (low_bound + up_bound) / 2 + (low_bound + up_bound) ./ (2 * k) - x(i, :) / k;
            
            % 边界处理
            opposite_x = Bounds(opposite_x, lb, ub);
            opposite_fit = fobj(opposite_x', func_num);
            
            % 贪婪选择 (保留更优解)
            if opposite_fit < fit(i)
                x(i, :) = opposite_x;
                fit(i) = opposite_fit;
            end
        end
        
        % ===== 更新个体历史最佳 =====
        for i = 1:pop
            if fit(i) < pFit(i)
                pFit(i) = fit(i);
                pX(i, :) = x(i, :);
            end
            
            if pFit(i) < fMin
                fMin = pFit(i);
                bestX = pX(i, :);
            end
        end
        
        Convergence_curve(t) = fMin;
        
        
    end
end

% ========== PWLCM混沌映射函数  ==========
function next_val = PWLCM(current_val)
    % 分段线性混沌映射(PWLCM)
    % 文献中p=0.4
    p = 0.4;
    
    if current_val >= 0 && current_val < p
        next_val = current_val / p;
    elseif current_val >= p && current_val < 0.5
        next_val = (current_val - p) / (0.5 - p);
    elseif current_val >= 0.5 && current_val < 1 - p
        next_val = (1 - current_val - p) / (0.5 - p);
    else
        next_val = (1 - current_val) / p;
    end
    
    % 确保值在[0,1]范围内
    next_val = max(0, min(1, next_val));
end

% ========== 边界处理函数 ==========
function s = Bounds(s, Lb, Ub)
    % 应用下界
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);
    
    % 应用上界
    J = temp > Ub;
    temp(J) = Ub(J);
    
    s = temp;
end
