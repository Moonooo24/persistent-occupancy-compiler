"""Automatic twin discovery and selective, recoverable data reuse.

The constructor is structural: no truth table or Hilbert-space array is used.
All public results preserve the prescribed product on the independent-set code.
"""
from collections import deque
from fractions import Fraction
from functools import lru_cache
import json
import math
from numbers import Integral

from runtime import QC, Parameter, G, core, io, sha, VENDOR


RULES_VERSION = 'journal_auto_reuse_v2_phase_safe'
KAPPA_MAX = 8


def graph_spec(case):
    if not isinstance(case['n'], Integral) or isinstance(case['n'], bool):
        raise ValueError('n must be an exact integer')
    n = int(case['n'])
    if n <= 0:
        raise ValueError('n must be positive')
    if any(len(e)!=2 or any(not isinstance(j,Integral) or isinstance(j,bool) for j in e) for e in case['graph_edges']):
        raise ValueError('Edges require two exact integer endpoints')
    edge_list = [tuple(sorted(map(int, e))) for e in case['graph_edges']]
    if len(set(edge_list)) != len(edge_list) or any(not (0 <= a < b < n) for a, b in edge_list):
        raise ValueError('Require a simple undirected graph with distinct edges')
    adj = [set() for _ in range(n)]
    for a, b in edge_list:
        adj[a].add(b)
        adj[b].add(a)
    signatures = {}
    for i in range(n):
        signatures.setdefault(tuple(sorted(adj[i] | {i})), []).append(i)
    parts = sorted(signatures.values(), key=lambda c: c[0])
    membership = {i: k for k, c in enumerate(parts) for i in c}
    quotient = []
    for a, c in enumerate(parts):
        assert all(j in adj[i] for p, i in enumerate(c) for j in c[p+1:])
        for b in range(a+1, len(parts)):
            d = parts[b]
            count = sum(j in adj[i] for i in c for j in d)
            assert count in (0, len(c)*len(d)), 'Closed-neighborhood classes must be modules'
            if count:
                quotient.append([a, b])
    order = list(case.get('target_order', range(n)))
    if any(not isinstance(i,Integral) or isinstance(i,bool) for i in order):
        raise ValueError('Target order requires exact integer labels')
    if sorted(order) != list(range(n)):
        raise ValueError('Target order must be a permutation of all original data')
    weights = [Fraction(str(w)) for w in case.get('weights', [1]*n)]
    if len(weights) != n:
        raise ValueError('One exact weight per original data bit is required')
    return dict(n=n, adj=adj, parts=parts, membership=membership,
                quotient=quotient, order=order, weights=weights)


def gf2_transform(rows, mask):
    answer = 0
    while mask:
        bit = mask & -mask
        answer ^= rows[bit.bit_length()-1]
        mask ^= bit
    return answer


def gf2_inverse(rows):
    n = len(rows)
    a, inv = list(rows), [1 << i for i in range(n)]
    for i in range(n):
        pivot = next(j for j in range(i, n) if a[j] >> i & 1)
        a[i], a[pivot] = a[pivot], a[i]
        inv[i], inv[pivot] = inv[pivot], inv[i]
        for j in range(n):
            if j != i and (a[j] >> i & 1):
                a[j] ^= a[i]
                inv[j] ^= inv[i]
    assert a == [1 << i for i in range(n)]
    assert all(gf2_transform(rows, v) == 1 << i for i, v in enumerate(inv))
    return inv


def mask_of(items):
    return sum(1 << i for i in items)


def supports(spec, i):
    own = spec['parts'][spec['membership'][i]]
    rest = [j for j in own if j != i]
    external = [c for c in spec['parts'] if c != own and c[0] in spec['adj'][i]]
    blocks = ([rest + external[0]] + external[1:]) if rest and external else ([rest] if rest else []) + external
    blocks = sorted([sorted(c) for c in blocks], key=lambda c: (len(c), tuple(c)))
    flat = [j for c in blocks for j in c]
    assert len(set(flat)) == len(flat) and set(flat) == spec['adj'][i]
    assert all(j in spec['adj'][a] for c in blocks for z, a in enumerate(c) for j in c[z+1:])
    return blocks


def encoding(spec, cached):
    pairs = [[j, c[0]] for k, c in enumerate(spec['parts']) if k in cached for j in c[1:]]
    rows = [1 << i for i in range(spec['n'])]
    for a, b in pairs:
        rows[b] ^= rows[a]
    return pairs, rows


def prepare(spec, i, cached, e_rows):
    k = spec['membership'][i]
    c = spec['parts'][k]
    pairs = ([[i, c[0]]] if i != c[0] else [[j, i] for j in c[1:]]) if k in cached else []
    rows = list(e_rows)
    for a, b in pairs:
        rows[b] ^= rows[a]
    assert rows[i] == 1 << i and all((v >> i & 1) == int(j == i) for j, v in enumerate(rows))
    inv = gf2_inverse(rows)
    blocks = supports(spec, i)
    factors = [gf2_transform(inv, mask_of(c)) for c in blocks]
    assert all(f and not (f >> i & 1) for f in factors)
    assert all(not (a & b) for k, a in enumerate(factors) for b in factors[k+1:])
    assert all(gf2_transform(rows, f) == mask_of(c) for f, c in zip(factors, blocks))
    return pairs, rows, inv, blocks, factors


def parity_network(n, factors):
    pairs, pivots = [], []
    for mask in factors:
        wires = [j for j in range(n) if mask >> j & 1]
        pivot = min(wires)
        pivots.append(pivot)
        pairs.extend([j, pivot] for j in wires if j != pivot)
    return pairs, sorted(pivots)


@lru_cache(maxsize=None)
def compact_template(family, degree):
    manifest = VENDOR / 'code/mis_phase_portable_v5/mis_phase_templates_manifest_v5.json'
    data = json.loads(manifest.read_text(encoding='utf8'))
    item = next(x for x in data['templates'] if x['family'] == family and x['degree'] == degree)
    path = manifest.parent / item['file']
    assert sha(path) == item['sha256']
    with path.open('rb') as f:
        original = io.load(f)[0]
    used = sorted({original.find_bit(w).index for op in original.data for w in op.qubits})
    active = list(range(degree+1)) + [j for j in used if j > degree]
    compact = QC(len(active))
    compact.global_phase = original.global_phase
    for ins in original.data:
        compact.append(ins.operation.copy(), [active.index(original.find_bit(w).index) for w in ins.qubits])
    assert core.costs(compact)['cx'] == item['cost']['cx']
    return compact, dict(family=family, degree=degree, source_sha256=item['sha256'],
                         source_file=str(path.relative_to(VENDOR)), active_indices=active,
                         unused_padding_removed=sorted(set(range(original.num_qubits))-set(active)),
                         scope=item['scope'], arbitrary_dirty_restoration=True)


def exact_kernel(n, target, controls, beta):
    q = QC(n)
    if not controls:
        q.rx(2*beta, target)
        return q, dict(kind='uncontrolled', degree=0)
    # This pinned Qiskit no-ancilla primitive is exact on all data states.
    positive = core.positive_mcx(len(controls), 'noaux')
    assert positive.num_qubits == len(controls)+1
    for j in controls:
        q.x(j)
    q.h(target)
    q.rz(beta, target)
    q.compose(positive, controls+[target], inplace=True)
    q.rz(-beta, target)
    q.compose(positive.inverse(), controls+[target], inplace=True)
    q.h(target)
    for j in controls:
        q.x(j)
    return q, dict(kind='exact_noaux_fallback', degree=len(controls),
                   primitive='pinned Qiskit synth_mcx_noaux_v24 plus exact two-query rotation',
                   whole_Hilbert_space=True, no_extra_data=True)


def query_kernel(n, target, controls, beta, mode):
    rank = len(controls)
    if not rank:
        return exact_kernel(n, target, controls, beta)
    family = 'qibo_vale_symbolic_rx' if mode == 't3' else 'conditional_phase_mixed'
    if rank <= KAPPA_MAX:
        template, cert = compact_template(family, rank)
        extra = template.num_qubits-rank-1
        available = [j for j in range(n) if j != target and j not in controls]
        if extra <= len(available):
            mapping = [target] + list(controls) + available[:extra]
            q = QC(n)
            q.compose(template.assign_parameters({p: beta for p in template.parameters}), mapping, inplace=True)
            return q, dict(cert, physical_mapping=mapping, arbitrary_borrowed_original_data=available[:extra],
                           care='target0 arbitrary controls; target1 only controls0' if mode != 't3' else 'global exact NOR RX(2beta)')
    q, cert = exact_kernel(n, target, controls, beta)
    cert['fallback_reason'] = 'outside frozen bank or insufficient original dirty data'
    return q, cert


def macro(spec, i, cached, e_rows, beta, route='phi'):
    n = spec['n']
    pre, rows, inv, blocks, factors = prepare(spec, i, cached, e_rows)
    q = QC(n)
    for a, b in pre:
        q.cx(a, b)
    rank = len(factors)
    info = dict(target=i, original_neighbors=sorted(spec['adj'][i]), blocks=blocks, rank=rank,
                prepare_CX=pre, actual_inverse_prepare_CX=list(reversed(pre)),
                GF2_rows=rows, GF2_inverse=inv, factors=factors,
                target_row_column_isolated=True, disjoint_factors=True,
                Feas_to_query_care=True, input_zero_assumed=False)
    if route in ('phi','walsh') and 1 <= rank <= (3 if route == 'phi' else 8):
        current, path = 0, []
        q.h(i)
        for k in range(1 << rank):
            gray = k ^ (k >> 1)
            mask = 0
            for j, factor in enumerate(factors):
                if gray >> j & 1:
                    mask ^= factor
            for j in range(n):
                if (current ^ mask) >> j & 1:
                    q.cx(j, i)
            q.rz(2*beta/float(1 << rank), i)
            current = mask
            path.append(mask)
        q.h(i)
        end_original = gf2_transform(rows, current)
        assert not (end_original & ~mask_of(spec['adj'][i]))
        info.update(query='open_Walsh', path=path, end_mask=current, end_original=end_original,
                    output_phase_fixed_on_Feas=True)
    else:
        pp, pivots = parity_network(n, factors)
        for a, b in pp:
            q.cx(a, b)
        kernel, cert = query_kernel(n, i, pivots, beta, 't3' if route == 't3' else 'phi')
        q.compose(kernel, inplace=True)
        for a, b in reversed(pp):
            q.cx(a, b)
        info.update(query='compressed_care_query' if route != 't3' else 'fixed_T3_Vale',
                    parity_CX=pp, actual_inverse_parity_CX=list(reversed(pp)), pivots=pivots,
                    kernel=cert, highrank_Walsh_terms_constructed=0)
    for a, b in reversed(pre):
        q.cx(a, b)
    return q, info


def weighted_cost(spec, cached, gamma, completion='clique'):
    n = spec['n']
    q, cert = QC(n), []
    if completion == 'original':
        enc, _ = encoding(spec, cached)
        for a, b in reversed(enc):
            q.cx(a, b)
        coeff = spec['weights']
        for i, w in enumerate(coeff):
            if w:
                q.rz(gamma*float(w), i)
                q.global_phase += gamma*float(w)/2
        for a, b in enc:
            q.cx(a, b)
        return q, dict(mode='actual_E_inverse_original_cost_E', weights=[str(w) for w in coeff])
    if completion != 'clique':
        raise ValueError(completion)
    for k, c in enumerate(spec['parts']):
        anchor, wa = c[0], spec['weights'][c[0]]
        pp = [[j, anchor] for j in c[1:]] if k not in cached else []
        if wa:
            for a, b in pp:
                q.cx(a, b)
            q.rz(gamma*float(wa), anchor)
            q.global_phase += gamma*float(wa)/2
            for a, b in reversed(pp):
                q.cx(a, b)
        coeff = [(j, spec['weights'][j]-wa) for j in c[1:]]
        for j, w in coeff:
            if w:
                q.rz(gamma*float(w), j)
                q.global_phase += gamma*float(w)/2
        cert.append(dict(clique=c, cached=k in cached, anchor_weight=str(wa),
                         leaf_coefficients=[[j,str(w)] for j,w in coeff], parity_compute_CX=pp))
    return q, dict(mode='same_original_coordinate_clique_parity_completion', cliques=cert,
                   exact_rational_weights=[str(w) for w in spec['weights']], cost_is_negative_weighted_occupancy=True)


def build(case, *, cache='all', route='phi', sweeps=None, cost_completion='clique'):
    spec = graph_spec(case)
    p = sweeps if sweeps is not None else case.get('sweeps', 2)
    if not isinstance(p,Integral) or isinstance(p,bool):
        raise ValueError('Sweep count must be an exact integer')
    p = int(p)
    if route not in ('phi','walsh','t3'):
        raise ValueError('Unknown query route')
    if cost_completion not in ('clique','original'):
        raise ValueError('Unknown cost completion')
    if p < 1:
        raise ValueError('At least one sweep required')
    eligible = {k for k,c in enumerate(spec['parts']) if len(c)>1}
    if cache == 'all':
        cached = eligible
    elif cache == 'none':
        cached = set()
    else:
        ids = list(cache)
        if any(not isinstance(k,Integral) or isinstance(k,bool) for k in ids):
            raise ValueError('Cache module indices must be exact integers')
        cached = set(map(int,ids))
    if not cached <= eligible:
        raise ValueError('Only nontrivial discovered cliques can be cached')
    if route == 't3' and cached:
        raise ValueError('Fixed T3 baseline uses local affine maps only, not OURS global cache')
    order = spec['order']
    orders = case.get('sweep_orders', [order]*p)
    if len(orders) != p or any(any(not isinstance(i,Integral) or isinstance(i,bool) for i in x) or sorted(x) != list(range(spec['n'])) for x in orders):
        raise ValueError('Prescribed sweep orders must all be complete permutations')
    beta = [Parameter('beta'+str(k)) for k in range(p)]
    gamma = [Parameter('gamma'+str(k)) for k in range(p-1)]
    local_beta = Parameter('local_beta')
    pairs, e_rows = encoding(spec, cached)
    local, certificates = {}, {}
    for i in range(spec['n']):
        local[i], certificates[i] = macro(spec, i, cached, e_rows, local_beta, route)
    out = QC(spec['n'])
    for a,b in pairs:
        out.cx(a,b)
    cost_certs = []
    for k in range(p):
        for i in orders[k]:
            out.compose(local[i].assign_parameters({local_beta:beta[k]}), inplace=True)
        if k < p-1:
            qc, cc = weighted_cost(spec, cached, gamma[k], cost_completion)
            out.compose(qc, inplace=True)
            cost_certs.append(cc)
    for a,b in reversed(pairs):
        out.cx(a,b)
    plan = dict(schema=RULES_VERSION, n=spec['n'], discovered_cliques=spec['parts'],
                quotient_edges=spec['quotient'], automatic_partition=True,
                cached_clique_indices=sorted(cached), encoding_CX=pairs,
                actual_inverse_encoding_CX=list(reversed(pairs)), sweeps=p,
                sweep_orders=orders, cost_layers=p-1, cost_completion=cost_completion,
                weights=[str(w) for w in spec['weights']], cost_certificates=cost_certs,
                macros=[certificates[i] for i in range(spec['n'])],
                zero_added_work_qubits=True, physical_qubits=spec['n'],
                exact_complex_semantics='Canonical prescribed product on original Feas, including global phase; not a global-canonical claim',
                structural_all_Feas_contract=True, enumerated_states=0,
                lowrank_rule='rank<=3 open Walsh, rank>=4 frozen care kernel with exact fallback',
                route=route, raw_metrics=core.costs(out))
    return out, plan


def topology_distances(n, topology=None):
    if topology in (None,'all_to_all'):
        return [[int(i != j) for j in range(n)] for i in range(n)]
    if topology == 'line':
        return [[abs(i-j) for j in range(n)] for i in range(n)]
    if topology == 'grid':
        a = max(k for k in range(1,math.isqrt(n)+1) if n % k == 0)
        b = n//a
        return [[abs(i//b-j//b)+abs(i%b-j%b) for j in range(n)] for i in range(n)]
    raise ValueError(topology)


def cnot_proxy(q, distances):
    return sum(distances[q.find_bit(op.qubits[0]).index][q.find_bit(op.qubits[1]).index]
               for op in q.data if op.operation.name == 'cx')


def source_proxy(case, cached, distances=None, route='phi', cost_completion='clique'):
    """Exact additive source-CX/hop objective without GF2 inverse or circuits."""
    spec = graph_spec(case)
    n, parts = spec['n'], spec['parts']
    p = case.get('sweeps', 2)
    if not isinstance(p,Integral) or isinstance(p,bool) or p<1:
        raise ValueError('Sweep count must be a positive exact integer')
    if route not in ('phi','walsh','t3') or cost_completion not in ('clique','original'):
        raise ValueError('Unknown route or cost completion')
    if any(not isinstance(k,Integral) or isinstance(k,bool) for k in cached):
        raise ValueError('Cache module indices must be exact integers')
    if not set(cached)<={k for k,c in enumerate(parts) if len(c)>1}:
        raise ValueError('Invalid cached module')
    distances = distances or topology_distances(n)
    score = lambda pairs: sum(distances[a][b] for a,b in pairs)
    pairs, _ = encoding(spec, cached)
    total = 2*score(pairs)
    beta = Parameter('proxy_beta')
    for i in range(n):
        ck = spec['membership'][i]
        own = parts[ck]
        pre = ([[i,own[0]]] if i != own[0] else [[j,i] for j in own[1:]]) if ck in cached else []
        factors = []
        for block in supports(spec, i):
            present = set(block)
            support = []
            for k,c in enumerate(parts):
                intersection = present.intersection(c)
                if not intersection:
                    continue
                if k == ck:
                    support.extend([own[0]] if k in cached and i != own[0] else sorted(intersection))
                else:
                    assert intersection == set(c)
                    support.extend([c[0]] if k in cached else c)
            factors.append(mask_of(support))
        rank = len(factors)
        local = 2*score(pre)
        if route in ('phi','walsh') and 1 <= rank <= (3 if route=='phi' else 8):
            current = 0
            for k in range(1 << rank):
                gray=k^(k>>1);mask=0
                for j,f in enumerate(factors):
                    if gray>>j&1:mask^=f
                local += sum(distances[j][i] for j in range(n) if (current^mask)>>j&1)
                current=mask
        elif rank:
            pp,pivots=parity_network(n,factors)
            kernel,_=query_kernel(n,i,pivots,beta,'t3' if route=='t3' else 'phi')
            local += 2*score(pp)+cnot_proxy(kernel,distances)
        total += p*local
    if cost_completion=='clique':
        for k,c in enumerate(parts):
            if k not in cached and spec['weights'][c[0]]:
                total += 2*(p-1)*score([[j,c[0]] for j in c[1:]])
    elif cost_completion=='original':
        total += 2*(p-1)*score(pairs)
    else:
        raise ValueError(cost_completion)
    return int(total)


def select_cache(case, topology=None):
    """Exact minimizer of the separable fixed-recipe source-CX/hop proxy.

    Native simplification and actual routing are outside this restricted optimum.
    The anchor, factor order, pivots, kernel and original layout stay fixed.
    """
    spec=graph_spec(case)
    eligible=[k for k,c in enumerate(spec['parts']) if len(c)>1]
    distances=topology_distances(spec['n'],topology)
    base=source_proxy(case,set(),distances)
    marginal=[]
    for k in eligible:
        value=source_proxy(case,{k},distances)
        marginal.append(dict(clique_index=k,clique=spec['parts'][k],delta=value-base))
    selected={x['clique_index'] for x in marginal if x['delta']<0}
    actual=source_proxy(case,selected,distances)
    predicted=base+sum(x['delta'] for x in marginal if x['delta']<0)
    assert actual==predicted,'Violation of fixed-recipe additive-cache theorem'
    return selected,dict(objective='source CX count' if topology in (None,'all_to_all') else 'sum hop distances per source CX in fixed layout',
                         baseline_proxy=base,selected_clique_indices=sorted(selected),final_proxy=actual,
                         marginal_terms=marginal,affine_sum_verified=True,
                         exact_optimum_scope='All subsets of discovered nontrivial modules, fixed anchors/block order/query recipe, before native cancellation and routing',
                         zero_marginal_tie='leave uncached',not_native_or_routed_optimality=True,
                         no_external_test_labels_used=True)

# V2 engineering/numerical adapter: no change to original frozen source.
_cost_before_phase_fix = weighted_cost
def weighted_cost(spec,cached,gamma,completion='clique'):
    from phase_safe import normalize_owned_phase
    q,cert=_cost_before_phase_fix(spec,cached,gamma,completion)
    q,phasecert=normalize_owned_phase(q)
    cert['affine_phase_certificate']=phasecert
    return q,cert

_query_before_generic = query_kernel
def query_kernel(n,target,controls,beta,mode):
    if mode!='t3' and len(controls)>KAPPA_MAX:
        from generic_query_v2 import kernel
        return kernel(n,target,controls,beta,exact=False)
    return _query_before_generic(n,target,controls,beta,mode)

from phase_safe import native as _phase_safe_native
core.native=_phase_safe_native
