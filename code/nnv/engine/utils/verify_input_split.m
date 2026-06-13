function status = verify_input_split(net, lb, ub, prop, reachOpt, max_depth, t_start, t_budget)
% VERIFY_INPUT_SPLIT  Sound input-domain branch-and-bound HOLDS-prover.
%
%   Proves the property HOLDS over the input box [lb,ub] by recursively
%   bisecting the WIDEST input dimension and verifying each sub-box with a
%   sound over-approximation (reachOpt, e.g. approx-star). The full box HOLDS
%   iff EVERY covering sub-box HOLDS; if any sub-box cannot be proven HOLDS
%   within the depth/time budget, the result is UNKNOWN.
%
%   SOUNDNESS. The two sub-boxes partition [lb,ub] along dimension j, so every
%   input x in [lb,ub] lies in one sub-box; hence net([lb,ub]) is contained in
%   the union of the sub-box images. If a sound over-approximation of each
%   sub-box image avoids the unsafe region, then net([lb,ub]) avoids it too.
%   No feasible point is dropped: this only ever proves HOLDS or returns
%   UNKNOWN. (The violated/SAT direction is handled by falsification upstream;
%   over-approx reach never returns SAT.) Splitting can only TIGHTEN bounds, so
%   a box that an over-approx already proves safe is never re-opened.
%
%   Inputs:
%     net       - NN object (must support net.reach(Star, reachOpt))
%     lb, ub    - flat input box bounds (column or row vectors)
%     prop      - property (cell/struct as consumed by verify_specification)
%     reachOpt  - reach options struct (e.g. struct('reachMethod','approx-star'))
%     max_depth - max bisection depth (e.g. 10)
%     t_start   - tic handle marking the start of the per-instance budget
%     t_budget  - wall-clock budget in seconds; UNKNOWN once exceeded
%
%   Output:
%     status    - 1 = holds (robust/unsat), 2 = unknown
%
%   This is the sound width-killer for benchmarks whose looseness comes from
%   the input-box width (e.g. lsnc_relu's bilinear x'Rx), where exact-star
%   explodes and a single approx-star pass is too loose.

    lb = lb(:); ub = ub(:);

    % time budget guard
    if toc(t_start) > t_budget
        status = 2; return;
    end

    % verify the current box with the sound over-approx method
    status = local_verify_box(net, lb, ub, prop, reachOpt);
    if status == 1
        return;                 % this box is provably safe
    end

    % inconclusive on this box -> bisect if depth remains
    if max_depth <= 0
        status = 2; return;
    end
    w = ub - lb;
    [wmax, j] = max(w);
    if wmax <= 0                 % degenerate (point) box that didn't verify
        status = 2; return;
    end
    mid = 0.5 * (lb(j) + ub(j));

    % lower half [lb, mid] on dim j
    ub1 = ub; ub1(j) = mid;
    s1 = verify_input_split(net, lb, ub1, prop, reachOpt, max_depth - 1, t_start, t_budget);
    if s1 ~= 1
        status = 2; return;      % a covering sub-box is not provably safe
    end
    % upper half [mid, ub] on dim j
    lb2 = lb; lb2(j) = mid;
    status = verify_input_split(net, lb2, ub, prop, reachOpt, max_depth - 1, t_start, t_budget);
    % HOLDS iff both halves hold (s1 == 1 already checked)
end

function status = local_verify_box(net, lb, ub, prop, reachOpt)
    % Sound over-approx verification of a single box. Returns 1 (holds) only if
    % the over-approx reach set provably avoids the unsafe region; otherwise 2
    % (unknown) -- a reach error is mapped to unknown (claims nothing).
    try
        IS = Star(lb, ub);
        ySet = net.reach(IS, reachOpt);
        s = verify_specification(ySet, prop);
    catch
        s = 2;
    end
    if s == 1
        status = 1;
    else
        status = 2;             % normalize: only HOLDS is conclusive here
    end
end
