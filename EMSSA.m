function [fMin, bestX, Convergence_curve] = EMSSA (pop, MaxIter, lb, ub, dim, fobj, func_num)
% 改进麻雀搜索算法 (EMSSA)：融合 UDOS、HATS、DES 三大策略
P_percent = 0.2; % 生产者比例
pNum = round (pop * P_percent); % 生产者数量
SD_percent = 0.1; % 危险感知者比例
sdNum = round (pop * SD_percent);% 危险感知者数量
ST = 0.8; % 安全阈值
delta = 1; % UDOS 调整参数
lambda = 1.6; % HATS 调节因子常数
mu = 1; % HATS 权重因子常数
alpha_hats = 2; % HATS 生长函数常数
tau = 7000; % DES 相似性参数
% 初始化种群（UDOS 策略：自适应帐篷混沌）
[x, status_value] = UDOS_init (pop, dim, lb, ub, delta);
fit = feval (fobj, x', func_num);
% 初始化个体最优和全局最优
pX = x;
pFit = fit;
[fMin, bestIdx] = min (fit);
bestX = x (bestIdx, :);
Convergence_curve = zeros (1, MaxIter);
% 主迭代循环
for t = 1:MaxIter
% 1. 种群排序与角色分配
[~, sortIdx] = sort (pFit);
producerIdx = sortIdx (1:pNum); % 生产者索引
scroungerIdx = setdiff (1:pop, producerIdx); % 清道夫索引
dangerIdx = randsample (1:pop, sdNum); % 危险感知者索引（随机选择）
% 2. 生产者位置更新（SSA 原有公式）
for i = 1:length (producerIdx)
idx = producerIdx (i);
R = rand ();
alpha = rand ();
if R < ST
x (idx, :) = pX (idx, :) .* exp (-i / (alpha * MaxIter));
else
x (idx, :) = pX (idx, :) + randn (1, dim) .* ones (1, dim);
end
x (idx, :) = Bounds (x (idx, :), lb, ub);
end
% 3. 跟随者位置更新（SSA 原有公式）
[~, bestProIdx] = min(pFit(producerIdx));
bestProX = pX(producerIdx(bestProIdx), :);
[fmax, worstIdx] = max(pFit);
worstX = pX(worstIdx, :);
for i = 1:length(scroungerIdx)
    idx = scroungerIdx(i);
    A = floor(rand(1, dim)*2)*2 - 1; % 生成1×dim的行向量（元素为1或-1）
    if idx > pop/2
        x(idx, :) = randn(1) * exp((worstX - pX(idx, :))/idx^2);
    else
        % 计算A*A'，增加奇异矩阵防护（避免逆矩阵不存在）
        A_A = A * A';
        if abs(A_A) < 1e-10 % 若A*A'接近0，改用随机扰动更新
            x(idx, :) = bestProX + randn(1, dim) .* abs(pX(idx, :) - bestProX);
        else
            % 修正：将 .* 改为 *（矩阵乘法），维度匹配：1×dim * dim×dim = 1×dim
            x(idx, :) = bestProX + abs(pX(idx, :) - bestProX) * (A' * inv(A_A)) * ones(1, dim);
        end
    end
    x(idx, :) = Bounds(x(idx, :), lb, ub);
end

% 4. 危险感知者位置更新（HATS 策略）
[fitBest, bestGIdx] = min (pFit);
bestGX = pX (bestGIdx, :);
[fitWorst, worstGIdx] = max (pFit);
worstGX = pX (worstGIdx, :);
for i = 1:length (dangerIdx)
idx = dangerIdx (i);
% 计算调节因子 rf（生长函数 Pearle 模型）
rf = lambda / (1 + exp ((alpha_hats * t)^2 / MaxIter^2));
% 计算权重因子 wf
wf = alpha_hats * (1 - mu * t / MaxIter) + 1;
% 根据适应度选择更新方式
r2 = rand () * 2 * pi;
r3 = rand () * 2;
if pFit (idx) ~= fitBest
x (idx, :) = wf * bestGX + rf * sin (r2) * abs (r3 * pX (idx, :) - bestGX);
else
x (idx, :) = wf * pX (idx, :) + rf * cos (r2) * abs (r3 * pX (idx, :) - worstGX);
end
x (idx, :) = Bounds (x (idx, :), lb, ub);
end
% 5. 动态进化策略（DES：最优个体相似扰动）
[currentFMin, currentBestIdx] = min (fit);
currentBestX = x (currentBestIdx, :);
% 计算扰动奇点
ub_vec = ones (1, dim) * ub;
lb_vec = ones (1, dim) * lb;
p_s = (ub_vec + lb_vec) / 2;
% 构建扰动函数（二维简化，高维逐维处理）
for d = 1:dim
% 假设当前最优个体在 d 维的坐标 x2，y 方向偏移 d（文献示意图）
x2 = currentBestX (d);
dy = 0.1; % 偏移量（文献未明确，设为小常数）
% 计算斜率 k
if abs (x2 - p_s (d)) < 1e-10
k = 0;
else
k = dy / (x2 - p_s (d));
end
% 扰动函数（垂直于奇点 - 最优个体连线）
if abs (dy) < 1e-10
fp = zeros (1, dim);
else
fp (d) = (ub_vec (d) + lb_vec (d) - 2 * x2) / (2 * dy) * x2 + ...
( - (ub_vec (d) + lb_vec (d))^2 + 2 * x2 * (ub_vec (d) + lb_vec (d)) ) / (4 * dy);
end
% 生成相似个体 x1
x1 = ((tau + 1) * (ub_vec (d) + lb_vec (d)) - 2 * x2) / (2 * tau);
% 构建相似个体
similarX = currentBestX;
similarX (d) = x1;
similarX = Bounds (similarX, lb, ub);
similarFit = feval (fobj, similarX', func_num);
% 比较更新
if similarFit < currentFMin
currentBestX = similarX;
currentFMin = similarFit;
end
end
% 更新全局最优
if currentFMin < fMin
fMin = currentFMin;
bestX = currentBestX;
x (currentBestIdx, :) = currentBestX;
fit (currentBestIdx) = currentFMin;
end
% 6. 适应度更新与个体最优更新
fit_new = feval (fobj, x', func_num);
for i = 1:pop
if fit_new (i) < pFit (i)
pFit (i) = fit_new (i);
pX (i, :) = x (i, :);
end
if pFit (i) < fMin
fMin = pFit (i);
bestX = pX (i, :);
end
end
fit = fit_new;
% 记录收敛曲线
Convergence_curve (t) = fMin;
end
end
% -------------------------------------------------------------------------
% 辅助函数 1：UDOS 初始化（自适应帐篷混沌）
% -------------------------------------------------------------------------
function [x, status_value] = UDOS_init (pop, dim, lb, ub, delta)
Max_iter = pop * dim; % 混沌映射迭代次数（种群规模×维度）
status_value = rand (1, Max_iter); % 初始映射值（0-1）
% 自适应帐篷混沌映射
for i = 1:Max_iter-1
    if status_value (i) == 0
        status_value (i+1) = 1 - delta * rand ();
    elseif status_value (i) > 0 && status_value (i) < 0.5
        status_value (i+1) = 1 - 2 * status_value (i);
    else
        status_value (i+1) = 2 * (1 - status_value (i));
    end
end
% 反向映射到解空间（修正 popdim 为 pop*dim）
status_value = status_value (1:pop*dim); % 取前 pop×dim 个混沌变量
x = reshape(status_value, pop, dim); % 重塑为 pop 行（个体）、dim 列（维度）
x = x .* (ub - lb) + lb; % 映射到 [lb, ub] 解空间
end
% -------------------------------------------------------------------------
% 辅助函数 2：边界处理
% -------------------------------------------------------------------------
function s = Bounds (s, Lb, Ub)
s = max (min (s, Ub), Lb);
end
