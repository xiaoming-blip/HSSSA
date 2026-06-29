function [fMin, bestX, Convergence_curve] = ISSA(pop, M, c, d, dim, fobj, func_num)
    P_percent = 0.2;    % Producers比例
    SD_percent = 0.2;   % 初始危险感知者比例
    ST = 0.8;           % 安全阈值
    % 均匀设计参数
    n_prime = nextprime(pop); % 大于pop的最小质数
    H_n = find(gcd(1:n_prime-1, n_prime) == 1); % 与n_prime互质的整数集合
    % 生成均匀设计点
    UD = zeros(n_prime, dim);
    for j = 1:length(H_n)
        h = H_n(j);
        for i = 1:n_prime
            UD(i, j) = mod(i * h, n_prime);
        end
    end
    UD = UD / n_prime; % 缩放至[0,1]
    % 随机选择pop行和dim列
    idx = randperm(n_prime, pop);
    UD = UD(idx, 1:dim);
    % 映射到搜索空间
    lb = c .* ones(1, dim);
    ub = d .* ones(1, dim);
    x = lb + (ub - lb) .* UD;
    % 评估初始种群
    fit = zeros(1, pop);
    for i = 1:pop
        fit(i) = fobj(x(i, :)', func_num);
    end
    pFit = fit;
    pX = x;
    [fMin, bestI] = min(fit);
    bestX = x(bestI, :);
    
    % 初始化WD相关参数
    theta = [0.3, 0.5, 0.7]; % 均匀性关联阈值初始值
    gamma_prev = 0; % 上一代全局最优适应度
    Convergence_curve = zeros(1, M);
    
    for t = 1:M
        % 计算当前种群均匀性WD
        wd_val = WD(x);
        phi = 1 - wd_val / sqrt((4/3)^dim - (3/2)^dim + 1); % 归一化均匀性度量
        % 确定均匀性状态
        if phi < theta(1)
            A_phi = 1; % Low
        elseif phi < theta(2)
            A_phi = 2; % Medium
        else
            A_phi = 3; % High
        end
        % 计算全局最优相对变化
        gamma = abs(fMin - gamma_prev) / (abs(fMin) + 1e-50);
        gamma_prev = fMin;
        % 调整theta based on gamma
        if gamma < 0.01
            theta = theta + 0.1; % 右移
        elseif gamma >= 0.1
            theta = theta - 0.1; % 左移
        end
        theta = max(min(theta, 1), 0); % 限制在[0,1]范围内
        % 根据均匀性状态调整危险感知者比例
        if A_phi == 1
            SD = 0.1;
        elseif A_phi == 2
            SD = 0.15;
        else
            SD = 0.2;
        end
        SD_num = round(pop * SD);
        
        [~, sortIndex] = sort(pFit);
        [fmax, B] = max(pFit);
        worse = x(B, :);
        
        % 更新生产者
        for i = 1:round(pop * P_percent)
            r2 = rand();
            if r2 < ST
                % 无危险：新公式
                beta1 = rand();
                beta2 = rand();
                r1 = rand();
                x(sortIndex(i), :) = pX(sortIndex(i), :) + beta1 * (rand(1, dim) - 0.5) * exp(-(i) / (r1 * M));
            else
                % 有危险：新公式
                beta2 = rand();
                % 选择前50%适应度的随机个体
                idx_rand = sortIndex(1:round(pop * 0.5));
                X_rand = x(idx_rand(randi(length(idx_rand))), :);
                x(sortIndex(i), :) = pX(sortIndex(i), :) + beta2 * (X_rand - pX(sortIndex(i), :));
            end
            x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub, phi, A_phi, bestX, worse); % 修改的边界处理
            fit(sortIndex(i)) = fobj(x(sortIndex(i), :)', func_num);
        end
        
        [fMMin, bestII] = min(fit);
        bestXX = x(bestII, :);
        
        % 更新scroungers
        for i = (round(pop * P_percent) + 1):pop
            A = floor(rand(1, dim) * 2) * 2 - 1;
            if i > (pop / 2)
                x(sortIndex(i), :) = randn(1, dim) .* exp((worse - pX(sortIndex(i), :)) / (i)^2);
            else
                x(sortIndex(i), :) = bestXX + abs(pX(sortIndex(i), :) - bestXX) * (A' * (A * A')^(-1)) * ones(1, dim);
            end
            x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub, phi, A_phi, bestX, worse); % 修改的边界处理
            fit(sortIndex(i)) = fobj(x(sortIndex(i), :)', func_num);
        end
        
        % 更新危险感知者
        for j = 1:SD_num
            if fit(sortIndex(j)) > fMin
                x(sortIndex(j), :) = bestX + randn(1, dim) .* abs(pX(sortIndex(j), :) - bestX);
            else
                x(sortIndex(j), :) = pX(sortIndex(j), :) + (2 * rand(1) - 1) * abs(pX(sortIndex(j), :) - worse) / (pFit(sortIndex(j)) - fmax + 1e-50);
            end
            x(sortIndex(j), :) = Bounds(x(sortIndex(j), :), lb, ub, phi, A_phi, bestX, worse); % 修改的边界处理
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

function s = Bounds(s, Lb, Ub, phi, A_phi, bestX, worstX)
    % 基于种群均匀性的边界处理
    temp = s;
    I = temp < Lb;
    J = temp > Ub;
    if any(I) || any(J)
        if A_phi == 1 % Low uniformity
            % 设置到全局最差点附近
            temp(I | J) = worstX(I | J) + rand(size(find(I | J))) .* (bestX(I | J) - worstX(I | J));
        elseif A_phi == 2 % Medium
            % 随机设置在全局最优和最差点之间
            temp(I | J) = bestX(I | J) + rand(size(find(I | J))) .* (worstX(I | J) - bestX(I | J));
        else % High
            % 设置到全局最优附近
            temp(I | J) = bestX(I | J) + randn(size(find(I | J))) .* 0.1 .* (Ub(I | J) - Lb(I | J));
        end
    end
    s = temp;
end

function wd = WD(X)
    % 计算环绕L2差异（WD）
    n = size(X, 1);
    dim = size(X, 2);
    wd = 0;
    for i = 1:n
        for j = 1:n
            term = 1;
            for k = 1:dim
                term = term * (1.5 - abs(X(i, k) - X(j, k)) + abs(abs(X(i, k) - X(j, k)) - 0.5));
            end
            wd = wd + term;
        end
    end
    wd = sqrt(-(4/3)^dim + (wd / n^2));
end
