function [fMin, bestX, Convergence_curve] = RSSA(pop, M, c, d, dim, fobj,func_num)
    % RSSA: 动态步长反向学习麻雀搜索算法
    % 输入参数:
    %   pop - 种群大小
    %   M - 最大迭代次数
    %   c - 下界
    %   d - 上界
    %   dim - 维度
    %   fobj - 目标函数
    %
    % 输出参数:
    %   fMin - 最小适应度值
    %   bestX - 最佳位置
    %   Convergence_curve - 收敛曲线

    % 算法参数设置
    PD = round(0.2 * pop);     % 发现者数量
    SD = round(0.2 * pop);    % 警戒者数量
    Pc = 0.5;                  % 疯狂概率
    
    % 初始化边界
    lb = c .* ones(1, dim);
    ub = d .* ones(1, dim);
    
    % 1. 使用好点集初始化种群
    x = goodPointSet(pop, dim, lb, ub);
    
    % 计算初始适应度
    fit = zeros(1, pop);
    for i = 1:pop
        fit(i) = fobj(x(i, :)',func_num);
    end
    
    % 初始化个体最佳和全局最佳
    pFit = fit;
    pX = x;
    [fMin, bestI] = min(fit);
    bestX = x(bestI, :);
    
    % 收敛曲线
    Convergence_curve = zeros(1, M);
    
    % 开始迭代
    for t = 1:M
        % 排序适应度
        [~, sortIndex] = sort(pFit);
        
        % 找出最差解
        [fMax, worstI] = max(pFit);
        worstX = x(worstI, :);
        
        % 2. 更新发现者位置（分段动态步长策略）
        R2 = rand();
        for i = 1:PD
            idx = sortIndex(i);
            alpha = rand();
            
            % 非线性递减因子
            omega = 1 - (t/M)^2;
            
            % 分段动态步长更新公式
            if R2 < 0.8
                x(idx, :) = pX(idx, :) .* exp(-(i) / (alpha * M));
            else
                x(idx, :) = pX(idx, :) + randn(1, dim);
            end
            
            % 应用非线性递减因子
            x(idx, :) = x(idx, :) * omega;
            
            % 边界检查
            x(idx, :) = BoundCheck(x(idx, :), lb, ub);
            
            % 更新适应度
            fit(idx) = fobj(x(idx, :)',func_num);
        end
        
        % 3. 更新跟随者位置（疯狂算子策略）
        for i = (PD + 1):pop
            idx = sortIndex(i);
            
            % 原始跟随者更新
            if i > pop/2
                x(idx, :) = randn(1, dim) .* exp((worstX - pX(idx, :)) / (i^2));
            else
                A = floor(rand(1, dim) * 2) * 2 - 1;
                A_plus = A' * (A * A')^(-1);
                x(idx, :) = bestX + abs(pX(idx, :) - bestX) * A_plus * ones(1, dim);
            end
            
            % 应用疯狂算子
            if rand() < Pc
                Q_prime = sign(rand() - 0.5);
                craziness = 0.0001 * Q_prime;
                x(idx, :) = x(idx, :) + craziness;
            end
            
            % 边界检查
            x(idx, :) = BoundCheck(x(idx, :), lb, ub);
            
            % 更新适应度
            fit(idx) = fobj(x(idx, :)',func_num);
        end
        
        % 4. 更新警戒者位置
        for i = 1:SD
            idx = sortIndex(randi(pop));
            
            beta = randn();
            f_i = fit(idx);
            
            if f_i > fMin
                x(idx, :) = bestX + beta .* abs(pX(idx, :) - bestX);
            else
                f_w = fMax;
                if rand > 0.5
                    direction = 1;
                else
                    direction = -1;
                end
                k = unidrnd(dim) * direction;
                x(idx, :) = pX(idx, :) + k .* abs(pX(idx, :) - worstX) / (f_i - f_w + realmin);
            end
            
            % 边界检查
            x(idx, :) = BoundCheck(x(idx, :), lb, ub);
            
            % 更新适应度
            fit(idx) = fobj(x(idx, :)',func_num);
        end
        
        % 5. t分布反向学习
        for i = 1:pop
            % 生成t分布反向解
            x_reverse = tDistributionOpposition(x(i, :), lb, ub, t, M);
            fit_reverse = fobj(x_reverse',func_num);
            
            % 如果反向解更优，则替换
            if fit_reverse < fit(i)
                x(i, :) = x_reverse;
                fit(i) = fit_reverse;
            end
        end
        
        % 更新个体最佳和全局最佳
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
        
        % 记录收敛曲线
        Convergence_curve(t) = fMin;
        
       
    end
end

function x = goodPointSet(pop, dim, lb, ub)
    % 好点集初始化种群
    % 公式: x_ij = lb_j + r_i * (ub_j - lb_j)
    % 其中r_i是好点集序列
    
    x = zeros(pop, dim);
    
    % 生成好点集序列
    r = zeros(pop, 1);
    for i = 1:pop
        r(i) = mod(i * sqrt(3), 1);
    end
    
    % 应用好点集初始化
    for i = 1:pop
        for j = 1:dim
            x(i, j) = lb(j) + r(i) * (ub(j) - lb(j));
        end
    end
end

function x_reverse = tDistributionOpposition(x, lb, ub, t, M)
    % t分布反向学习
    % 公式: x_reverse = lb + ub - x + t(t) * (rand() - 0.5)
    
    % 生成t分布随机数（以迭代次数为自由度）
    t_noise = trnd(t, size(x));
    
    % 计算反向解
    x_reverse = lb + ub - x + t_noise .* (rand(size(x)) - 0.5);
    
    % 边界检查
    x_reverse = BoundCheck(x_reverse, lb, ub);
end

function s = BoundCheck(s, Lb, Ub)
    % 边界检查函数
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);
    
    J = temp > Ub;
    temp(J) = Ub(J);
    
    s = temp;
end
