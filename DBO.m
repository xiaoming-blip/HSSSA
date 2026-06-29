function [fMin, bestX, Convergence_curve] = DBO(pop, MaxIter, lb, ub, dim, fobj,func_num)
% 文献来源：Dung Beetle Optimizer (2023, Journal of Supercomputing)
% 核心参数设置（严格匹配文献Table 1）
k = 0.1;     % 偏转系数
b = 0.3;     % 光强系数
S = 0.5;     % 偷窃行为系数
roll_num = round(pop * 0.2);   
brood_num = round(pop * 0.2);   
forage_num = round(pop * 0.25); 
thief_num = pop - roll_num - brood_num - forage_num;

% 初始化种群
if isscalar(lb), lb = lb * ones(1, dim); elseif iscolumn(lb), lb = lb(:)'; end
if isscalar(ub), ub = ub * ones(1, dim); elseif iscolumn(ub), ub = ub(:)'; end
x = lb + (ub - lb) .* rand(pop, dim);  % 种群位置初始化
fit = feval(fobj, x',func_num);                % 计算适应度

% 初始化最优解
pFit = fit;
pX = x;
[fMin, bestIdx] = min(fit);
bestX = x(bestIdx, :);
Convergence_curve = zeros(1, MaxIter);

% 初始化繁殖球和觅食区域
brood_pos = lb + (ub - lb) .* rand(brood_num, dim);  % 繁殖球初始位置
forage_Lb = lb;  % 觅食区域下界
forage_Ub = ub;  % 觅食区域上界

for t = 1:MaxIter
    % 1. 全局最优/最差更新
    [fBest, gBestIdx] = min(pFit);
    globalBest = pX(gBestIdx, :);
    [fWorst, gWorstIdx] = max(pFit);
    globalWorst = pX(gWorstIdx, :);
    
    % 2. 滚球蜣螂位置更新（公式1-2）
    for i = 1:roll_num
        idx = i;
        % 直线滚球（公式1）
        if rand(1) < 0.9
            alpha = 1;
            if rand(1) > 0.5, alpha = -1; end
            Delta_x = abs(x(idx, :) - globalWorst);
            x(idx, :) = x(idx, :) + alpha * k * x(idx, :) + b * Delta_x;
        else
            % 跳舞转向（公式2）
            theta = rand(1) * pi;
            if theta ~= 0 && theta ~= pi/2 && theta ~= pi
                x(idx, :) = x(idx, :) + tan(theta) * abs(x(idx, :) - pX(idx, :));
            end
        end
        x(idx, :) = Bounds_DBO(x(idx, :), lb, ub);
    end
    
    % 3. 繁殖球位置更新（公式3-4）
    R = 1 - t / MaxIter;
    X_star = globalBest;
    Lb_star = max(X_star .* (1 - R), lb);
    Ub_star = min(X_star .* (1 + R), ub);
    for i = 1:brood_num
        idx = roll_num + i;
        b1 = rand(1, dim);
        b2 = rand(1, dim);
        brood_pos(i, :) = X_star + b1 .* (brood_pos(i, :) - Lb_star) + b2 .* (brood_pos(i, :) - Ub_star);
        brood_pos(i, :) = Bounds_DBO(brood_pos(i, :), Lb_star, Ub_star);
        % 繁殖球融入种群
        x(idx, :) = brood_pos(i, :);
    end
    
    % 4. 觅食蜣螂位置更新（公式6）
    for i = 1:forage_num
        idx = roll_num + brood_num + i;
        C1 = randn(1);
        C2 = rand(1, dim);
        x(idx, :) = globalBest + C1 .* (x(idx, :) - forage_Lb) + C2 .* (x(idx, :) - forage_Ub);
        x(idx, :) = Bounds_DBO(x(idx, :), forage_Lb, forage_Ub);
    end
    
    % 5. 偷窃蜣螂位置更新（公式7）
    for i = 1:thief_num
        idx = roll_num + brood_num + forage_num + i;
        g = randn(1, dim);
        x(idx, :) = globalBest + g .* S .* abs(x(idx, :) - globalBest);
        x(idx, :) = Bounds_DBO(x(idx, :), lb, ub);
    end
    
    % 6. 适应度更新与最优解更新
    fit_new = feval(fobj, x',func_num);
    for i = 1:pop
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
    
    % 记录收敛曲线
    Convergence_curve(t) = fMin;
end
end

% 辅助函数：边界控制（文献默认策略）
function s = Bounds_DBO(s, Lb, Ub)
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);
    J = temp > Ub;
    temp(J) = Ub(J);
    s = temp;
end
