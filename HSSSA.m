function [fMin,bestX, Convergence_curve] = HSSSA(pop, MaxIter, lb, ub, dim, fobj,func_num)

    % -------- 参数设置 --------
    P_percent = 0.2;
    pNum = round(pop * P_percent);

    % HMM / Q 学习参数
    numStates = 3;          
    numObservations = 3;    
    numActions = 3;         

    % Q 学习参数
    alpha = 0.12;
    gamma = 0.92;

    % ε-贪心策略参数
    epsilon_init = 0.5;
    epsilon_min = 0.1;
    epsilon_decay = 0.998;
    epsilon = epsilon_init;

    % Baum-Welch 参数
    BW_UPDATE_INTERVAL = 5;
    BW_max_iter = 10;
    BW_epsilon = 1e-4;

    % 滑动窗口参数
    WINDOW_SIZE = 10;
    improvement_window = [];

    % 奖励机制参数
    c1 = 100;
    c2 = 50;
    delta_penalty = 0.01;
    epsilon_div = 1e-10;

    % 景观检测参数
    LANDSCAPE_CHECK_INTERVAL = 20;
    MIN_REBUILD_INTERVAL = 30;

    % 邻居机制参数
    NUM_FIRST_HOP = 3;
    NUM_SECOND_HOP = 2;

    % -------- 初始化种群 --------
    x = lb + (ub - lb) .* rand(pop, dim);
    % ⭐ 修改：逐个评估适应度
    fit = zeros(pop, 1);
    for i = 1:pop
        fit(i) = feval(fobj, x(i, :)',func_num);
    end

    pX = x;
    pFit = fit;
    [fMin, bestIdx] = min(fit);
    bestX = x(bestIdx, :);
    Convergence_curve = zeros(1, MaxIter);

    % 历史记录
    pX_prev = pX;
    pFit_prev = pFit;

    % -------- HMM 初始化 --------
    pi_hmm = ones(1, numStates) / numStates;
    
    B_hmm = [0.40, 0.35, 0.25;  
             0.20, 0.50, 0.30;  
             0.10, 0.25, 0.65]; 

   A_action = ones(numStates, numActions, numStates) / numStates;

    Q_table = zeros(numStates, numActions) + 50;

    % -------- 缓冲区 --------
    belief_state = pi_hmm;
    prev_belief = pi_hmm;
    prev_fMin = fMin;
    last_action = 1;
    last_state = 1;

    % ⭐ 添加维特比算法所需变量
    delta = pi_hmm;  % 维特比算法的delta变量
    psi = zeros(1, numStates);  % 维特比算法的psi变量（路径回溯）

    history.fitness_history = zeros(1, MaxIter+1);
    history.fitness_history(1) = fMin;

    last_network_rebuild_iter = 0;
    last_landscape_type = 'unknown';

    % 统计结构体
    Statistics.state_sequence = zeros(1, MaxIter);
    Statistics.action_sequence = zeros(1, MaxIter);
    Statistics.reward_sequence = zeros(1, MaxIter);
    Statistics.Q_history = zeros(MaxIter, numStates, numActions);
    Statistics.Num_A = zeros(1, numActions);
    Statistics.improvement_history = zeros(1, MaxIter);
    Statistics.mu_history = zeros(1, MaxIter);
    Statistics.sigma_history = zeros(1, MaxIter);
    Statistics.epsilon_history = zeros(1, MaxIter);
    Statistics.observation_history = zeros(1, MaxIter);
    Statistics.belief_history = zeros(MaxIter, numStates);
    Statistics.global_imp_rate_history = zeros(1, MaxIter);
    Statistics.phi_history = zeros(1, MaxIter);
    Statistics.BW_iterations = zeros(1, MaxIter);

    observation_buffer = [];
    action_buffer = [];

    Statistics.pi_history = zeros(MaxIter, numStates);
    Statistics.A_action_history = cell(MaxIter, 1);
    Statistics.B_history = cell(MaxIter, 1);

    % 初始化网络
    selectionProb = calculateSelectionProbability(pX, pFit, dim, 1, MaxIter);
    [hubNodes, adjacencyMatrix] = buildScaleFreeNetwork_WithAdjacency(selectionProb, pNum, pop);

    % ================ 主循环 ===============
    for t = 1:MaxIter
        pX_prev = pX;
        pFit_prev = pFit;
        
        % ========== 步骤0：景观检测与网络重构 ==========
        if mod(t, LANDSCAPE_CHECK_INTERVAL) == 0 && (t - last_network_rebuild_iter) >= MIN_REBUILD_INTERVAL
            theta = calculateFitnessLandscapeTheta(pX, pFit);
            Statistics.phi_history(t) = theta; 
            
            if t > LANDSCAPE_CHECK_INTERVAL
                prev_check_idx = find(Statistics.phi_history(1:t-1) > 0, 1, 'last');
                if isempty(prev_check_idx)
                    prev_theta = 0;
                else
                    prev_theta = Statistics.phi_history(prev_check_idx);
                end
                
                delta_theta = abs(theta - prev_theta);
                
                if (prev_theta == 0 && theta > 0) || (prev_theta > 0 && theta == 0) || delta_theta > 0.2
                    selectionProb = calculateSelectionProbability(pX, pFit, dim, t, MaxIter);
                    [hubNodes, adjacencyMatrix] = buildScaleFreeNetwork_WithAdjacency(selectionProb, pNum, pop);
                    last_network_rebuild_iter = t;
                    if theta == 0
                        last_landscape_type = 'Unimodal_Strict';
                    else
                        last_landscape_type = 'Multimodal_Rugged';
                    end
                end
            end
        end

        % ========== 步骤1：SSA基础更新 ==========
        [~, sortIdx] = sort(pFit);
        [fmax, idxWorst] = max(pFit);
        worse = x(idxWorst, :);
        discoverIdx = hubNodes;

        r2 = rand();
        if r2 < 0.8
            for i = 1:length(discoverIdx)
                idx = discoverIdx(i);
                [firstHop, secondHop] = getTwoHopNeighbors(idx, pX, pFit, pX_prev, pFit_prev, adjacencyMatrix, NUM_FIRST_HOP, NUM_SECOND_HOP, t);
                
                if ~isempty(firstHop)
                    avgFirstHopPos = mean(pX(firstHop, :), 1);
                else
                    avgFirstHopPos = pX(idx, :);
                end
                
                if ~isempty(secondHop)
                    bestSecondHopPos = pX(secondHop, :);
                else
                    bestSecondHopPos = avgFirstHopPos;
                end
                
                r1 = rand();
                alpha_decay = exp(-(i) / (r1 * MaxIter));
                x(idx, :) = pX(idx, :) * alpha_decay + (1 - alpha_decay) * (0.5 * bestX + 0.3 * avgFirstHopPos + 0.2 * bestSecondHopPos);
                x(idx, :) = Bounds(x(idx, :), lb, ub);
                fit(idx) = feval(fobj, x(idx, :)',func_num);
            end
        else
            for i = 1:length(discoverIdx)
                idx = discoverIdx(i);
                [firstHop, secondHop] = getTwoHopNeighbors(idx, pX, pFit, pX_prev, pFit_prev, adjacencyMatrix, NUM_FIRST_HOP, NUM_SECOND_HOP, t);
                
                if ~isempty(firstHop)
                    [~, bestFirstIdx] = min(pFit(firstHop));
                    bestFirstHopPos = pX(firstHop(bestFirstIdx), :);
                else
                    bestFirstHopPos = pX(idx, :);
                end
                
                if ~isempty(secondHop)
                    bestSecondHopPos = pX(secondHop, :);
                else
                    bestSecondHopPos = bestFirstHopPos;
                end
                
                x(idx, :) = pX(idx, :) + 0.3 * (bestFirstHopPos - pX(idx, :)) + 0.2 * (bestSecondHopPos - pX(idx, :));
                x(idx, :) = Bounds(x(idx, :), lb, ub);
                fit(idx) = feval(fobj, x(idx, :)',func_num);
            end
        end

        [fMMin, bestII] = min(fit);
        bestXX = x(bestII, :);

        allIndices = 1:pop;
        followerIndices = setdiff(allIndices, discoverIdx);
        [~, followerOrder] = ismember(followerIndices, sortIdx);
        [~, reorder] = sort(followerOrder);
        followerIndices = followerIndices(reorder);

        for i = 1:length(followerIndices)
            idx = followerIndices(i);
            A = floor(rand(1, dim) * 2) * 2 - 1;
            if idx > (pop/2)
                x(idx, :) = randn(1) * exp((worse - pX(idx, :)) / (idx)^2);
            else
                if rcond(A * A') > 1e-10
                    x(idx, :) = bestXX + (abs((pX(idx, :) - bestXX))) * (A' * inv(A * A')) * ones(1, dim);
                else
                    x(idx, :) = bestXX + rand(1, dim) .* (pX(idx, :) - bestXX);
                end
            end
            x(idx, :) = Bounds(x(idx, :), lb, ub);
            fit(idx) = feval(fobj, x(idx, :)',func_num);
        end

        c = randperm(pop);
        b = c(1:min(10, pop));
        for j = 1:length(b)
            idx = b(j);
            if pFit(idx) > fMin
                x(idx, :) = bestX + (randn(1, dim)).* (abs((pX(idx, :) - bestX)));
            else
                x(idx, :) = pX(idx, :) + (2 * rand(1) - 1) * (abs(pX(idx, :) - worse)) / (pFit(idx) - fmax + 1e-50);
            end
            x(idx, :) = Bounds(x(idx, :), lb, ub);
            fit(idx) = feval(fobj, x(idx, :)',func_num);
        end

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

        % ========== 步骤2：计算SSA更新后的改善情况 ==========
        if t == 1
            avg_improvement = 0;
            mu = 0;
            sigma = 1e-6;
            observation = 3;
        else
            individual_improvements = zeros(1, pop);
            num_improved = 0;
    
            for i = 1:pop
                if abs(pFit_prev(i)) > 1e-10
                    improvement = abs(pFit_prev(i) - pFit(i)) / abs(pFit_prev(i));
                    if pFit(i) < pFit_prev(i)
                        individual_improvements(i) = improvement;
                        num_improved = num_improved + 1;
                    end
                end
            end
    
            if num_improved > 0
                improved_values = individual_improvements(individual_improvements > 0);
                avg_improvement = mean(improved_values);
            else
                avg_improvement = 0;
            end
    
            improvement_window = [improvement_window, avg_improvement];
            if length(improvement_window) > WINDOW_SIZE
                improvement_window(1) = [];
            end
    
            if length(improvement_window) > 0
                mu = mean(improvement_window);
                sigma = std(improvement_window);
                if sigma < 1e-6
                    sigma = 1e-6;
                end
            else
                mu = 0;
                sigma = 1e-6;
            end
    
            observation = getObservationByStatistics(avg_improvement, mu, sigma);
        end
        
        Statistics.improvement_history(t) = avg_improvement;
        Statistics.mu_history(t) = mu;
        Statistics.sigma_history(t) = sigma;
        Statistics.observation_history(t) = observation;
        
        observation_buffer = [observation_buffer, observation];
        if t > 1
            action_buffer = [action_buffer, last_action];
        end

        % ========== 步骤3：HMM维特比算法状态估计 + 动作选择 ==========
        if t == 1
            % 初始化：δ_1(i) = π_i * b_i(X_1)
            delta = pi_hmm(:)' .* B_hmm(:, observation)';
            psi = zeros(1, numStates);
            [~, estimated_state] = max(delta);
        else
            % 递推：δ_t(j) = max_i [δ_{t-1}(i) * a_ij] * b_j(X_t)
            [delta, psi, estimated_state] = viterbiOnline_Update(delta, A_action, B_hmm, observation, last_action, numStates);
        end
        
        % 保存belief为delta的归一化版本（用于统计记录）
        if sum(delta) > 0
            belief_state = delta / sum(delta);
        else
            belief_state = ones(1, numStates) / numStates;
        end
        Statistics.belief_history(t, :) = belief_state;

        Q_values = Q_table(estimated_state, :);
        
        if rand() < epsilon
            action = randi(numActions);
        else
            [~, action] = max(Q_values);
        end
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay);
        Statistics.epsilon_history(t) = epsilon;
        Statistics.Num_A(action) = Statistics.Num_A(action) + 1;

        % ========== 步骤4：执行增强动作（三种DE变体）==========
        switch action
            case 1
                [x, fit, pX, pFit, fMin, bestX] = applyDE_current_to_rand_1(... 
                    x, fit, pX, pFit, fMin, bestX, pop, dim, lb, ub, fobj,func_num,  t, MaxIter);
            case 2
                [x, fit, pX, pFit, fMin, bestX] = applyDE_current_to_best_1(...
                    x, fit, pX, pFit, fMin, bestX, pop, dim, lb, ub, fobj, func_num, t, MaxIter);
            case 3
                [x, fit, pX, pFit, fMin, bestX] = applyDE_current_to_pbest_JADE(...
                    x, fit, pX, pFit, fMin, bestX, pop, dim, lb, ub, fobj, func_num, t, MaxIter, pNum,true);
        end

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

        % ========== 步骤5：Q学习更新 ==========
        Q_values_current_state = Q_table(estimated_state, :);
        sum_Q = sum(Q_values_current_state);
        
        if sum_Q > epsilon_div
            action_pref = Q_values_current_state / sum_Q;
        else
            action_pref = ones(1, numActions) / numActions;
        end
        
        action_pref_last = action_pref(last_action);

        diversity = std(pFit);
        if t > 1
            prev_diversity = std(pFit_prev);
        else
            prev_diversity = diversity;
        end
        diversity_change = (diversity - prev_diversity) / (prev_diversity + epsilon_div);

        global_imp_rate = 0; 
        if abs(prev_fMin) > epsilon_div
            if fMin < prev_fMin
                global_imp_rate = (prev_fMin - fMin) / (abs(prev_fMin) + epsilon_div);
                immediate_reward = action_pref_last * c1 * global_imp_rate;
            else
                immediate_reward = -action_pref_last * c2 * delta_penalty;
            end
        else
            immediate_reward = 0;
        end
        
        Statistics.global_imp_rate_history(t) = global_imp_rate;
        immediate_reward = max(-200, min(300, immediate_reward));
        Statistics.reward_sequence(t) = immediate_reward;
        
        current_Q = Q_table(last_state, last_action);
        max_next_Q = max(Q_table(estimated_state, :));
        td_target = immediate_reward + gamma * max_next_Q;
        td_error = td_target - current_Q;
        Q_table(last_state, last_action) = current_Q + alpha * td_error;
        Q_table(last_state, last_action) = max(0, min(500, Q_table(last_state, last_action)));

        % ========== 步骤6：Baum-Welch更新 ==========
        if mod(t, BW_UPDATE_INTERVAL) == 0 && length(observation_buffer) >= BW_UPDATE_INTERVAL && length(action_buffer) >= (BW_UPDATE_INTERVAL-1)
            [pi_hmm, A_action, B_hmm, bw_iter_count] = BaumWelch_Update_ActionDependent_WithConvergence(...  
                pi_hmm, A_action, B_hmm, observation_buffer, action_buffer, ...  
                numStates, numObservations, numActions, BW_max_iter, BW_epsilon);
            
            Statistics.BW_iterations(t) = bw_iter_count;
            Statistics.pi_history(t, :) = pi_hmm;
            Statistics.A_action_history{t} = A_action;
            Statistics.B_history{t} = B_hmm;
            
            observation_buffer = [];
            action_buffer = [];
        end

        last_action = action;
        last_state = estimated_state;
        prev_belief = belief_state;
        prev_fMin = fMin;

        Convergence_curve(t) = fMin;
        Statistics.state_sequence(t) = estimated_state;
        Statistics.action_sequence(t) = action;
        Statistics.Q_history(t, :, :) = Q_table;
        history.fitness_history(t+1) = fMin;
    end

    % ========== 最终统计 ==========
    Statistics.final_Q_table = Q_table;
    Statistics.final_belief = belief_state;
    Statistics.final_pi = pi_hmm;
    Statistics.final_A_action = A_action;
    Statistics.final_B = B_hmm;
    Statistics.history = history;
    
    Statistics.summary.avg_improvement = mean(Statistics.improvement_history(Statistics.improvement_history > 0));
    Statistics.summary.max_improvement = max(Statistics.improvement_history);
    Statistics.summary.avg_global_imp_rate = mean(Statistics.global_imp_rate_history(Statistics.global_imp_rate_history > 0));
    Statistics.summary.avg_reward = mean(Statistics.reward_sequence);
    Statistics.summary.avg_phi = mean(Statistics.phi_history(Statistics.phi_history > 0));
    Statistics.summary.avg_BW_iter = mean(Statistics.BW_iterations(Statistics.BW_iterations > 0));
    Statistics.summary.action_names = {'DE/current-to-rand/1 (平衡探索)', 'DE/current-to-best/1 (局部开发)', 'DE/current-to-pbest/1 (JADE停滞逃逸)'};
    Statistics.summary.state_names = {'S1-全局探索', 'S2-局部开发', 'S3-停滞'};
    Statistics.summary.observation_names = {'O1-大改善', 'O2-小改善', 'O3-无改善/退化'};
end

% ⭐ 修改DE函数，改为逐个评估
function [x, fit, pX, pFit, fMin, bestX] = applyDE_current_to_rand_1(x, fit, pX, pFit, fMin, bestX, pop, dim, lb, ub, fobj,func_num,t, MaxIter)
    progress = t / MaxIter;
    F = 0.5 + 0.3 * (1 - progress);
    CR = 0.9;
    
    idx_all = repmat(1:pop, pop, 1);
    mask = ~eye(pop);
    candidates = zeros(pop, pop-1);
    for i = 1:pop
        candidates(i, :) = idx_all(i, mask(i, :));
    end
    
    [~, perm_idx] = sort(rand(pop, pop-1), 2);
    
    r1 = candidates(sub2ind([pop, pop-1], (1:pop)', perm_idx(:, 1)));
    r2 = candidates(sub2ind([pop, pop-1], (1:pop)', perm_idx(:, 2)));
    r3 = candidates(sub2ind([pop, pop-1], (1:pop)', perm_idx(:, 3)));
    
    mutant = pX + F * (pX(r1, :) - pX) + F * (pX(r2, :) - pX(r3, :));
    mutant = max(min(mutant, ub), lb);
    
    crossover_mask = rand(pop, dim) < CR;
    jrand = randi(dim, pop, 1);
    linear_idx = sub2ind([pop, dim], (1:pop)', jrand);
    crossover_mask(linear_idx) = true;
    
    trial = pX;
    trial(crossover_mask) = mutant(crossover_mask);
    
    % ⭐ 修改：逐个评估
    trialFit = zeros(pop, 1);
    for i = 1:pop
        trialFit(i) = feval(fobj, trial(i, :)',func_num);
    end
    
    improved = trialFit < fit;
    x(improved, :) = trial(improved, :);
    fit(improved) = trialFit(improved);
    
    pImproved = trialFit < pFit;
    pFit(pImproved) = trialFit(pImproved);
    pX(pImproved, :) = trial(pImproved, :);
    
    [newMin, minIdx] = min(trialFit);
    if newMin < fMin
        fMin = newMin;
        bestX = trial(minIdx, :);
    end
end

function [x, fit, pX, pFit, fMin, bestX] = applyDE_current_to_best_1(x, fit, pX, pFit, fMin, bestX, pop, dim, lb, ub, fobj, func_num, t, MaxIter)
    progress = t / MaxIter;
    F = 0.5 * (1 - progress) + 0.3;
    CR = 0.8;
    
    idx_all = repmat(1:pop, pop, 1);
    mask = ~eye(pop);
    candidates = zeros(pop, pop-1);
    for i = 1:pop
        candidates(i, :) = idx_all(i, mask(i, :));
    end
    
    [~, perm_idx] = sort(rand(pop, pop-1), 2);
    r1 = candidates(sub2ind([pop, pop-1], (1:pop)', perm_idx(:, 1)));
    r2 = candidates(sub2ind([pop, pop-1], (1:pop)', perm_idx(:, 2)));
    
    bestX_matrix = repmat(bestX, pop, 1);
    mutant = pX + F * (bestX_matrix - pX) + F * (pX(r1, :) - pX(r2, :));
    mutant = max(min(mutant, ub), lb);
    
    crossover_mask = rand(pop, dim) < CR;
    jrand = randi(dim, pop, 1);
    linear_idx = sub2ind([pop, dim], (1:pop)', jrand);
    crossover_mask(linear_idx) = true;
    
    trial = pX;
    trial(crossover_mask) = mutant(crossover_mask);
    
    % ⭐ 修改：逐个评估
    trialFit = zeros(pop, 1);
    for i = 1:pop
        trialFit(i) = feval(fobj, trial(i, :)',func_num);
    end
    
    improved = trialFit < fit;
    x(improved, :) = trial(improved, :);
    fit(improved) = trialFit(improved);
    
    pImproved = trialFit < pFit;
    pFit(pImproved) = trialFit(pImproved);
    pX(pImproved, :) = trial(pImproved, :);
    
    [newMin, minIdx] = min(trialFit);
    if newMin < fMin
        fMin = newMin;
        bestX = trial(minIdx, :);
    end
end

function [x, fit, pX, pFit, fMin, bestX] = applyDE_current_to_pbest_JADE(... 
    x, fit, pX, pFit, fMin, bestX, pop, dim, lb, ub, fobj,func_num,  t, MaxIter, pNum,resetArchive)
    
    persistent archive;
    if (nargin > 14 && resetArchive) || t == 1
        archive = [];
    end
    
    progress = t / MaxIter;
    F = 0.5 + 0.3 * rand();
    CR = 0.9;
    
    [~, sortIdx] = sort(pFit);
    p_best_idx = sortIdx(randi(pNum));
    
    idx_all = repmat(1:pop, pop, 1);
    mask = ~eye(pop);
    candidates = zeros(pop, pop-1);
    for i = 1:pop
        candidates(i, :) = idx_all(i, mask(i, :));
    end
    
    [~, perm_idx] = sort(rand(pop, pop-1), 2);
    r1 = candidates(sub2ind([pop, pop-1], (1:pop)', perm_idx(:, 1)));
    
    if ~isempty(archive)
        arch_size = size(archive, 1);
        r2_idx = randi(arch_size, pop, 1);
        r2_matrix = archive(r2_idx, :);
    else
        r2 = candidates(sub2ind([pop, pop-1], (1:pop)', perm_idx(:, 2)));
        r2_matrix = pX(r2, :);
    end
    
    pbest_matrix = repmat(pX(p_best_idx, :), pop, 1);
    mutant = pX + F * (pbest_matrix - pX) + F * (pX(r1, :) - r2_matrix);
    mutant = max(min(mutant, ub), lb);
    
    crossover_mask = rand(pop, dim) < CR;
    jrand = randi(dim, pop, 1);
    linear_idx = sub2ind([pop, dim], (1:pop)', jrand);
    crossover_mask(linear_idx) = true;
    
    trial = pX;
    trial(crossover_mask) = mutant(crossover_mask);
    
    % ⭐ 修改：逐个评估
    trialFit = zeros(pop, 1);
    for i = 1:pop
        trialFit(i) = feval(fobj, trial(i, :)',func_num);
    end
    
    improved = trialFit < fit;
    x(improved, :) = trial(improved, :);
    fit(improved) = trialFit(improved);
    
    failed = ~improved;
    if any(failed)
        archive = [archive; pX(failed, :)];
        if size(archive, 1) > pop
            archive = archive(randperm(size(archive, 1), pop), :);
        end
    end
    
    pImproved = trialFit < pFit;
    pFit(pImproved) = trialFit(pImproved);
    pX(pImproved, :) = trial(pImproved, :);
    
    [newMin, minIdx] = min(trialFit);
    if newMin < fMin
        fMin = newMin;
        bestX = trial(minIdx, :);
    end
end

% ===========================================================================
% ⭐ 新增：维特比算法在线更新函数
% ===========================================================================
function [delta_new, psi_new, estimated_state] = viterbiOnline_Update(delta_prev, A_action, B, observation, prev_action, numStates)
    % 维特比算法的在线递推更新
    % 输入：
    %   delta_prev: 上一时刻的delta值 (1 x numStates)
    %   A_action: 动作依赖的状态转移矩阵 (numStates x numActions x numStates)
    %   B: 观测概率矩阵 (numStates x numObservations)
    %   observation: 当前观测
    %   prev_action: 上一时刻的动作
    %   numStates: 状态数量
    % 输出：
    %   delta_new: 当前时刻的delta值 (1 x numStates)
    %   psi_new: 当前时刻的最优前驱状态索引 (1 x numStates)
    %   estimated_state: 当前估计的最可能状态
    
    % 获取对应动作的转移矩阵: A_action(:, prev_action, :)
    % 结果为 numStates x numStates，其中 A_slice(i,j) = P(S_t=j | S_{t-1}=i, a_{t-1})
    A_slice = squeeze(A_action(:, prev_action, :));  % numStates x numStates
    
    delta_new = zeros(1, numStates);
    psi_new = zeros(1, numStates);
    
    % 对每个状态j，计算 δ_t(j) = max_i [δ_{t-1}(i) * a_ij] * b_j(X_t)
    for j = 1:numStates
        % 计算所有前驱状态i到状态j的路径概率
        % delta_prev(i) * A_slice(i, j)
        values = delta_prev(:)' .* A_slice(:, j)';  % 1 x numStates
        
        % 找到最大值及其对应的前驱状态
        [max_val, max_idx] = max(values);
        
        % δ_t(j) = max_val * b_j(observation)
        delta_new(j) = max_val * B(j, observation);
        
        % ψ_t(j) = argmax_i [δ_{t-1}(i) * a_ij]
        psi_new(j) = max_idx;
    end
    
    % 防止数值下溢
    if sum(delta_new) < 1e-300
        delta_new = delta_new + 1e-300;
    end
    
    % 当前最可能的状态
    [~, estimated_state] = max(delta_new);
end

% ===========================================================================
% 动作依赖的 Baum-Welch 算法
% ===========================================================================
function [pi_new, A_action_new, B_new, iter_count] = BaumWelch_Update_ActionDependent_WithConvergence(...       
    pi, A_action, B, observations, actions, numStates, numObservations, numActions, max_iter, epsilon_converge)
    
    T = length(observations);
    T_action = length(actions);
    
    pi_curr = pi;
    A_curr = A_action;
    B_curr = B;
    
    prev_logLik = -inf;
    iter_count = 0;
    
    obs_matrix = repmat(observations(:), 1, numObservations) == repmat(1:numObservations, T, 1);
    action_matrix = repmat(actions(:), 1, numActions) == repmat(1:numActions, T_action, 1);
    
    for iter = 1:max_iter
        iter_count = iter;
        
        alpha = zeros(T, numStates);
        scaling = zeros(1, T);
        
        alpha(1, :) = pi_curr .* B_curr(:, observations(1))';
        scaling(1) = sum(alpha(1, :));
        scaling(1) = max(scaling(1), 1e-300);
        alpha(1, :) = alpha(1, :) / scaling(1);
        
        for t_idx = 2:T
            if t_idx - 1 <= T_action
                a_prev = actions(t_idx - 1);
                A_slice = squeeze(A_curr(:, a_prev, :));
                alpha(t_idx, :) = (alpha(t_idx-1, :) * A_slice) .* B_curr(:, observations(t_idx))';
                
                scaling(t_idx) = sum(alpha(t_idx, :));
                if scaling(t_idx) < 1e-300
                    scaling(t_idx) = 1e-300;
                end
                alpha(t_idx, :) = alpha(t_idx, :) / scaling(t_idx);
            end
        end
        
        logLikelihood = sum(log(scaling));
        
        if iter > 1 && abs(logLikelihood - prev_logLik) < epsilon_converge
            break;
        end
        prev_logLik = logLikelihood;
        
        beta = zeros(T, numStates);
        beta(T, :) = 1;
        
        for t_idx = T-1:-1:1
            if t_idx <= T_action
                a_t = actions(t_idx);
                A_slice = squeeze(A_curr(:, a_t, :));
                beta(t_idx, :) = A_slice * (B_curr(:, observations(t_idx+1)) .* beta(t_idx+1, :)');
                
                if scaling(t_idx+1) > 1e-300
                    beta(t_idx, :) = beta(t_idx, :) / scaling(t_idx+1);
                end
            end
        end
        
        gamma = alpha .* beta;
        gamma_sum = sum(gamma, 2);
        gamma_sum(gamma_sum < 1e-300) = 1e-300;
        gamma = gamma ./ gamma_sum;
        
        xi = zeros(T_action, numStates, numActions, numStates);
        
        for a = 1:numActions
            action_idx = find(action_matrix(:, a));
            
            if ~isempty(action_idx)
                T_a = length(action_idx);
                A_slice = squeeze(A_curr(:, a, :));
                
                for k = 1:T_a
                    t_k = action_idx(k);
                    
                    if t_k + 1 <= T
                        beta_B = beta(t_k+1, :) .* B_curr(:, observations(t_k+1))';
                        
                        temp = (alpha(t_k, :)' * beta_B) .* A_slice;
                        
                        denom = sum(temp(:));
                        denom = max(denom, 1e-300);
                        xi(t_k, :, a, :) = temp / denom;
                    end
                end
            end
        end
        
        pi_curr = gamma(1, :);
        pi_curr = pi_curr / sum(pi_curr);
        
        for a = 1:numActions
            action_indices = find(action_matrix(:, a));
            
            if ~isempty(action_indices)
                denom = sum(gamma(action_indices, :), 1);
                denom(denom < 1e-300) = 1e-300;
                
                numer = squeeze(sum(xi(action_indices, :, a, :), 1));
                
                A_curr(:, a, :) = numer ./ denom';
            else
                A_curr(:, a, :) = 1 / numStates;
            end
        end
        
        A_reshaped = reshape(A_curr, numStates * numActions, numStates);
        row_sums = sum(A_reshaped, 2);
        
        invalid_rows = row_sums < 1e-300;
        row_sums(invalid_rows) = numStates;
        A_reshaped(invalid_rows, :) = 1;
        
        A_reshaped = A_reshaped ./ row_sums;
        
        A_curr = reshape(A_reshaped, numStates, numActions, numStates);
        
        denom = sum(gamma, 1)';
        denom(denom < 1e-300) = 1e-300;
        
        B_curr = gamma' * obs_matrix;
        
        B_curr = B_curr ./ denom;
        
        row_sums = sum(B_curr, 2);
        invalid_rows = row_sums < 1e-300;
        B_curr(invalid_rows, :) = 1 / numObservations;
        
        B_curr = B_curr ./ sum(B_curr, 2);
    end
    
    pi_new = pi_curr;
    A_action_new = A_curr;
    B_new = B_curr;
end

% ===========================================================================
function theta = calculateFitnessLandscapeTheta(pX, pFit)
    NP = length(pFit);
    if NP <= 3, theta = 0; return; end
    
    [~, bestIdx] = min(pFit);
    x_star = pX(bestIdx, :);
    
    distances = sqrt(sum((pX - x_star).^2, 2));
    
    [~, sortIdx] = sort(distances);
    sorted_fit = pFit(sortIdx);
    
    c = 0;
    for m = 2:(NP - 1)
        if sorted_fit(m) < sorted_fit(m-1) && sorted_fit(m) < sorted_fit(m+1)
            c = c + 1;
        end
    end
    
    theta = c / (NP - 2); 
end

function observation = getObservationByStatistics(delta_t, mu, sigma)
    % ⭐ 新的观测划分规则：
    % O1 (大改善): improvement > μ + σ
    % O2 (小改善): μ < improvement ≤ μ + σ
    % O3 (无改善/退化): improvement ≤ μ (包括 ≤0 的情况)
    
    if delta_t > mu + sigma
        observation = 1;  % O1: 大改善
    elseif delta_t > mu
        observation = 2;  % O2: 小改善
    else
        observation = 3;  % O3: 无改善/退化
    end
end

function [firstHopNeighbors, secondHopNeighbor] = getTwoHopNeighbors(idx, pX, pFit, pX_prev, pFit_prev, adjacencyMatrix, numFirstHop, numSecondHop, t)
    pop = size(pX, 1);
    networkNeighbors = find(adjacencyMatrix(idx, :) > 0);
    if isempty(networkNeighbors)
        [~, sortedByFit] = sort(pFit);
        networkNeighbors = sortedByFit(1:min(numFirstHop, pop-1));
        networkNeighbors = setdiff(networkNeighbors, idx);
    end
    firstHopNeighbors = networkNeighbors(1:min(numFirstHop, length(networkNeighbors)));
    
    if isempty(firstHopNeighbors)
        secondHopNeighbor = [];
        return;
    end
    
    scores = zeros(1, length(firstHopNeighbors));
    for j = 1:length(firstHopNeighbors)
        node_j = firstHopNeighbors(j);
        if t > 1
            gradient_j = pX(node_j, :) - pX_prev(node_j, :);
            gradient_magnitude = norm(gradient_j);
            fitness_improvement = pFit_prev(node_j) - pFit(node_j);
            if gradient_magnitude > 1e-10
                gradient_quality = max(0, fitness_improvement) / gradient_magnitude;
            else
                gradient_quality = 0;
            end
        else
            gradient_quality = 0;
        end
        fitness_score = 1 / (pFit(node_j) + 1e-10);
        scores(j) = 0.5 * gradient_quality + 0.5 * fitness_score;
    end
    if max(scores) > min(scores)
        scores = (scores - min(scores)) / (max(scores) - min(scores));
    end
    [~, bestIdx] = max(scores);
    secondHopNeighbor = firstHopNeighbors(bestIdx);
end

function prob = calculateSelectionProbability(positions, fitness, dim, t, MaxIter)
    pop = size(positions, 1);
    maxFit = max(fitness);
    minFit = min(fitness);
    if maxFit == minFit
        prob = ones(1, pop) / pop;
    else
        fitnessScore = (maxFit - fitness) / (maxFit - minFit);
        prob = fitnessScore' / sum(fitnessScore);  % ⭐ 转置确保行向量
    end
end

function [hubNodes, adjacencyMatrix] = buildScaleFreeNetwork_WithAdjacency(selectionProb, numHubs, pop)
    degree = zeros(1, pop);
    adjacencyMatrix = zeros(pop, pop);
    is_initialized = false(1, pop); 
    
    % ⭐ 确保 selectionProb 是行向量
    selectionProb = selectionProb(:)';  % 强制转换为行向量
    
    [~, initIdx] = sort(selectionProb, 'descend');
    m0 = 3; 
    initialNodes = initIdx(1:m0); 
    
    degree(initialNodes) = m0 - 1;
    is_initialized(initialNodes) = true;
    
    initial_subgraph = ones(m0, m0) - eye(m0); 
    adjacencyMatrix(initialNodes, initialNodes) = initial_subgraph;
    
    newNodes = find(~is_initialized); 
    for newNode = newNodes
        
        potential_mask = (degree > 0) & (1:pop ~= newNode); 
        connectProb = zeros(1, pop);
        
        % ⭐ 使用 find 索引代替逻辑索引，避免维度问题
        valid_indices = find(potential_mask);
        if ~isempty(valid_indices)
            connectProb(valid_indices) = selectionProb(valid_indices) .* (degree(valid_indices) + 1);
        end
        sum_connectProb = sum(connectProb);
        
        if sum_connectProb > 1e-10
            
            m = 2; 
            
            valid_idx = find(connectProb > 0);
            valid_prob = connectProb(valid_idx);
            
            if sum(valid_prob) > 1e-10
                valid_prob = valid_prob / sum(valid_prob); 
            else
                is_initialized(newNode) = true;
                continue; 
            end
            
            m_actual = min(m, length(valid_idx)); 
            connected = [];
            
            if m_actual > 0
                
                current_prob = valid_prob; 
                
                for k = 1:m_actual
                    
                    cumProb = cumsum(current_prob); 
                    r = rand(); 
                    
                    selected_pos_in_valid = find(r <= cumProb, 1, 'first');
                    
                    if ~isempty(selected_pos_in_valid)
                        
                        selected_node_idx = valid_idx(selected_pos_in_valid);
                        connected = [connected, selected_node_idx];
                        
                        current_prob(selected_pos_in_valid) = 0;
                        
                        sum_remaining = sum(current_prob);
                        if sum_remaining > 1e-10
                            current_prob = current_prob / sum_remaining;
                        else
                            break;
                        end
                    else
                        break;
                    end
                end
            end
            
            if ~isempty(connected)
                degree(connected) = degree(connected) + 1; 
                adjacencyMatrix(newNode, connected) = 1;   
                adjacencyMatrix(connected, newNode) = 1;
                degree(newNode) = length(connected);       
            end
        end
        is_initialized(newNode) = true; 
    end
    
    hubScore = degree .* (1 + 2 * selectionProb);
    [~, hubIdx] = sort(hubScore, 'descend'); 
    
    numHubs_to_select = min(numHubs, pop);
    hubNodes = hubIdx(1:numHubs_to_select);
end

function s = Bounds(s, Lb, Ub)
    s = max(min(s, Ub), Lb);
end
