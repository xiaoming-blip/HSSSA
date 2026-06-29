function [fMin, bestX, Convergence_curve] = HSSA(pop, M, c, d, dim, fobj, func_num)
    P_percent = 0.2;    % Producers比例
    SD_percent = 0.2;   % 危险感知者比例
    ST = 0.8;           % 安全阈值
    % 初始化种群
    lb = c .* ones(1, dim);
    ub = d .* ones(1, dim);
    x = zeros(pop, dim);
    for i = 1:pop
        x(i, :) = lb + (ub - lb) .* rand(1, dim);
    end
    fit = zeros(1, pop);
    for i = 1:pop
        fit(i) = fobj(x(i, :)', func_num);
    end
    pFit = fit;
    pX = x;
    [fMin, bestI] = min(fit);
    bestX = x(bestI, :);
    Convergence_curve = zeros(1, M);
    
    % LCM参数
    lt = 10; % 生命周期阈值
    ts = 0;  % 陷入局部最优的计数器
    ms = 0;  % 用于记录scrounger改善情况的计数器
    
    for t = 1:M
        [~, sortIndex] = sort(pFit);
        [fmax, B] = max(pFit);
        worse = x(B, :);
        
        % 更新生产者
        for i = 1:round(pop * P_percent)
            r2 = rand();
            if r2 < ST
                x(sortIndex(i), :) = pX(sortIndex(i), :) * exp(-(i) / (rand() * M));
            else
                x(sortIndex(i), :) = pX(sortIndex(i), :) + randn(1, dim);
            end
            x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub);
            fit(sortIndex(i)) = fobj(x(sortIndex(i), :)', func_num);
        end
        
        [fMMin, bestII] = min(fit);
        bestXX = x(bestII, :);
        
        % 生成虚拟个体（VIS）
        % 选择两个随机生产者
        r1 = randi(round(pop * P_percent));
        r2 = randi(round(pop * P_percent));
        X_r1 = pX(sortIndex(r1), :);
        X_r2 = pX(sortIndex(r2), :);
        beta = rand();
        VX_B = X_r1 + beta * (bestX - X_r1);
        VX_P = X_r2 + beta * (bestXX - X_r2);
        
        % 更新scroungers（HS）
        sr = pop - round(pop * P_percent); % scroungers数量
        for i = (round(pop * P_percent) + 1):pop
            % 使用虚拟个体更新
            xi1 = randn();
            xi2 = randn();
            x(sortIndex(i), :) = pX(sortIndex(i), :) + xi1 * (VX_B - pX(sortIndex(i), :)) + xi2 * (VX_P - pX(sortIndex(i), :));
            x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub);
            fit(sortIndex(i)) = fobj(x(sortIndex(i), :)', func_num);
            
            % LCM检查
            if fit(sortIndex(i)) >= fMin
                ms = ms + 1;
            end
        end
        
        % 应用LCM
        if ms >= sr
            ts = ts + 1;
            ms = 0;
        end
        if ts >= lt
            for i = (round(pop * P_percent) + 1):pop
                kappa = randn();
                phi = 1e-5;
                % 随机选择个体
                idx_rand = randi(pop);
                X_rand = x(idx_rand, :);
                norm_val = norm(X_rand - x(sortIndex(i), :)) + phi;
                x(sortIndex(i), :) = pX(sortIndex(i), :) + kappa * (X_rand - pX(sortIndex(i), :)) / norm_val;
                x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub);
                fit(sortIndex(i)) = fobj(x(sortIndex(i), :)', func_num);
            end
            ts = 0;
        end
        
        % 更新危险感知者
        SD_num = round(pop * SD_percent);
        for j = 1:SD_num
            if fit(sortIndex(j)) > fMin
                x(sortIndex(j), :) = bestX + randn(1, dim) .* abs(pX(sortIndex(j), :) - bestX);
            else
                x(sortIndex(j), :) = pX(sortIndex(j), :) + (2 * rand(1) - 1) * abs(pX(sortIndex(j), :) - worse) / (pFit(sortIndex(j)) - fmax + 1e-50);
            end
            x(sortIndex(j), :) = Bounds(x(sortIndex(j), :), lb, ub);
            fit(sortIndex(j)) = fobj(x(sortIndex(j), :)', func_num);
        end
        
        % 更新个体最优和全局最优
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

function s = Bounds(s, Lb, Ub)
    % 简单边界处理
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);
    J = temp > Ub;
    temp(J) = Ub(J);
    s = temp;
end
