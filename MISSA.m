function [fMin,bestX,  Convergence_curve] = MISSA(pop, maxIter, lb, ub, dim, fobj, func_num)
    % MISSA: 多策略集成麻雀搜索算法
    % 修复初始化函数问题
    
    % 算法参数设置
    PD = round(0.2 * pop);     % 发现者数量
    SD = round(0.2 * pop);    % 警戒者数量
    ST = 0.8;                 % 安全阈值
    NL = round(PD / 3);       % 领导者数量
    
    % 初始化种群（使用修复后的初始化函数）
    X = initialization_fixed(pop, dim, ub, lb);
    
    % 计算初始适应度
    fitness = zeros(1, pop);
    for i = 1:pop
        fitness(i) = fobj(X(i, :)', func_num);
    end
    
    % 初始化个体最佳和全局最佳
    pFit = fitness;
    pX = X;
    [fMin, bestIdx] = min(fitness);
    bestX = X(bestIdx, :);
    
    % 收敛曲线
    Convergence_curve = zeros(1, maxIter);
    
    % 开始迭代
    for t = 1:maxIter
        % 排序适应度
        [~, sortIndex] = sort(pFit);
        
        % 找出最差解
        [~, worstIdx] = max(pFit);
        worstX = X(worstIdx, :);
        
        % 1. ISS策略：更新发现者位置
        R2 = rand();
        for i = 1:PD
            idx = sortIndex(i);
            
            % 改进搜索策略(ISS) - 公式(8)和(9)
            r = rand();
            factor = (1 - (t/maxIter)^2) * (1 - r) + (t/maxIter)^2 * r;
            
            if R2 < ST
                % 安全状态下的更新
                X(idx, :) = pX(idx, :) * factor;
            else
                % 危险状态下的更新
                X(idx, :) = pX(idx, :) + randn(1, dim);
            end
            
            % 边界检查
            X(idx, :) = BoundCheck(X(idx, :), lb, ub);
            
            % 更新适应度
            fitness(idx) = fobj(X(idx, :)', func_num);
        end
        
        % 2. GFS策略：更新跟随者位置
        for i = (PD + 1):pop
            idx = sortIndex(i);
            
            % 群体跟随策略(GFS) - 公式(12)和(13)
            % 选择领导者
            k = 1 + mod(i, NL);
            leaderIdx = sortIndex(k);
            
            % 更新位置
            r1 = rand();
            r2 = rand();
            X(idx, :) = pX(idx, :) + 2 * r1 * cos(2 * pi * r2) * (pX(leaderIdx, :) - pX(idx, :));
            
            % 边界检查
            X(idx, :) = BoundCheck(X(idx, :), lb, ub);
            
            % 更新适应度
            fitness(idx) = fobj(X(idx, :)', func_num);
        end
        
        % 3. 更新警戒者位置
        for i = 1:SD
            idx = sortIndex(randi(pop));
            
            beta = randn();
            f_i = fitness(idx);
            
            if f_i > fMin
                % 向最优解靠近
                X(idx, :) = bestX + beta .* abs(pX(idx, :) - bestX);
            else
                % 随机扰动
                f_w = max(fitness);
                k = randi(dim) * (2*(rand() > 0.5) - 1); % 随机方向
                X(idx, :) = pX(idx, :) + k .* abs(pX(idx, :) - worstX) / (f_i - f_w + realmin);
            end
            
            % 边界检查
            X(idx, :) = BoundCheck(X(idx, :), lb, ub);
            
            % 更新适应度
            fitness(idx) = fobj(X(idx, :)', func_num);
        end
        
        % 4. ROBLS策略：随机反向学习
        for i = 1:pop
            % 随机反向学习策略(ROBLS) - 公式(14)
            r = rand();
            X_reverse = lb + ub - X(i, :) + r * (rand(1, dim) - 0.5);
            
            % 边界检查
            X_reverse = BoundCheck(X_reverse, lb, ub);
            
            % 计算反向解适应度
            fit_reverse = fobj(X_reverse', func_num);
            
            % 如果反向解更优，则替换
            if fit_reverse < fitness(i)
                X(i, :) = X_reverse;
                fitness(i) = fit_reverse;
            end
        end
        
        % 更新个体最佳和全局最佳
        for i = 1:pop
            if fitness(i) < pFit(i)
                pFit(i) = fitness(i);
                pX(i, :) = X(i, :);
            end
            
            if pFit(i) < fMin
                fMin = pFit(i);
                bestX = pX(i, :);
            end
        end
        
        % 记录收敛曲线
        Convergence_curve(t) = fMin;
        
    end
end

function X = initialization_fixed(pop, dim, ub, lb)
    % 支持标量和向量形式的边界参数
    
    % 检查边界参数维度
    if isscalar(lb) && isscalar(ub)
        % 如果lb和ub是标量，扩展到所有维度
        X = lb + (ub - lb) * rand(pop, dim);
    elseif length(lb) == 1 && length(ub) == 1
        % 如果lb和ub是单元素向量，扩展到所有维度
        X = lb + (ub - lb) * rand(pop, dim);
    else
        % 如果lb和ub是向量，逐个维度初始化
        X = zeros(pop, dim);
        for i = 1:pop
            for j = 1:dim
                X(i, j) = lb(j) + (ub(j) - lb(j)) * rand();
            end
        end
    end
end

function s = BoundCheck(s, Lb, Ub)
    % 边界检查函数
    % 支持标量和向量形式的边界参数
    
    temp = s;
    
    if isscalar(Lb) && isscalar(Ub)
        % 标量边界
        temp(temp < Lb) = Lb;
        temp(temp > Ub) = Ub;
    else
        % 向量边界
        for j = 1:size(s, 2)
            temp(temp(:, j) < Lb(j), j) = Lb(j);
            temp(temp(:, j) > Ub(j), j) = Ub(j);
        end
    end
    
    s = temp;
end
