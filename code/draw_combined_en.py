"""Redraw the manuscript's paired evaluations as three vector figures."""
from collections import defaultdict

from figure_style import (
    BLUE, INK, ORANGE, PURPLE, SLATE, TEAL, WIDTH, Line2D, group,
    np, panel, plt, read, save, saving, summed_saving, yzero,
)
from matplotlib.colors import TwoSlopeNorm
import json

EVAL = 'data/evaluation.json'
STATS = 'data/review_statistics.json'
EDGES = 'data/graph_edges.json'
ENTRY = 'data/linux_entry.json'
TRIALS = 'data/linux_trials.json'
evaluation = read('evaluation.json')
cohorts = evaluation['cohorts']
main = list({r['semantic_program_sha256']: r for r in cohorts['main144']['programs']}.values())
confirm = cohorts['new_confirmation72']['programs']
sweep = cohorts['sweep120']['programs']
public = cohorts['public24']['programs']
assert (len(main), len(confirm), len(sweep), len(public)) == (134, 72, 120, 24)

COHORTS = [(main, 'Main', BLUE, 'o'), (confirm, 'Confirmation', TEAL, '^')]
SPLIT_LABELS = ['Reference to no cache', 'No cache to complete output']
SPLIT_COLORS = [BLUE, '#66AD9E']
LEFT = .95
RIGHT = 3.93
PANEL_WIDTH = 2.23


def axes_inches(fig, x, y, width, height, letter, title):
    w, h = fig.get_size_inches()
    ax = fig.add_axes((x/w, y/h, width/w, height/h))
    panel(ax, letter, title)
    return ax


def equality(ax, low, high, log=False):
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set(xlim=(low, high), ylim=(low, high))
    ax.plot([low, high], [low, high], c=SLATE, ls='--', lw=.75, zorder=1)


def mechanism_and_structure():
    fig = plt.figure(figsize=(WIDTH, 5.84))
    a = axes_inches(fig, LEFT, 4.39, PANEL_WIDTH, 1.14, 'a', 'Matched ablations')
    b = axes_inches(fig, RIGHT, 4.39, PANEL_WIDTH, 1.14, 'b', 'Neighborhoods and control rank')
    c = axes_inches(fig, LEFT, 2.66, 5.19, .98, 'c', 'Decomposition of CX savings')
    d = axes_inches(fig, LEFT, .73, PANEL_WIDTH, 1.26, 'd', 'Two-vertex class coverage')
    e = axes_inches(fig, RIGHT, .73, PANEL_WIDTH, 1.26, 'e', 'Class size redistribution')
    pairings = [
        ('Closed to open', 'ref_cpg_closed', 'ref_cpg_open', BLUE, 'o'),
        ('Exact to phase', 'ref_matched_exact_clique', 'ours_none', PURPLE, 's'),
        ('Empty to full cache', 'ours_none', 'ours_all', TEAL, '^'),
    ]
    degrees = np.array([2, 4, 6, 8])
    yzero(a)
    for i, (label, ref, new, color, marker) in enumerate(pairings):
        observed = []
        for degree in degrees:
            rows = [r for r in main if r['degree'] == degree]
            metrics = [r['all_reference_metrics'] | r['all_candidate_metrics'] for r in rows]
            observed.append(np.array([100*(1-m[new]['cx']/m[ref]['cx']) for m in metrics]))
        middle = np.array([np.median(gains) for gains in observed])
        lower = np.array([gains.min() for gains in observed])
        upper = np.array([gains.max() for gains in observed])
        a.errorbar(degrees + (i-1)*.17, middle, yerr=[middle-lower, upper-middle],
                   fmt=marker, ms=3.8, c=color, elinewidth=.8, capsize=1.7, label=label)
    a.set(xticks=degrees, xlabel='Quotient-graph degree', ylabel='Paired CX reduction (%)', ylim=(-3, 58))
    a.legend(loc='upper left', ncol=2, fontsize=7.1, handlelength=.85,
             handletextpad=.3, columnspacing=.6, borderaxespad=.23)

    edges = read('graph_edges.json')
    by_graph = group(main, 'graph_sha256')
    assert len(by_graph) == 62
    for pattern, label, color, marker in [('two', 'Two-vertex classes', BLUE, 'o'), ('mixed', 'Mixed sizes', ORANGE, '^')]:
        observed = [rows[0] for rows in by_graph.values() if rows[0]['pattern'] == pattern]
        max_degrees = []
        for r in observed:
            g = edges[r['case']]
            endpoints = [u for edge in g['edges'] for u in edge]
            max_degrees.append(np.bincount(endpoints, minlength=g['n']).max())
        b.scatter(max_degrees, [r['control_rank_max'] for r in observed],
                  s=20, c=color, marker=marker, linewidths=0, alpha=.62, label=label)
    b.plot([0, 10], [0, 10], ls='--', c=SLATE, lw=.75)
    b.set(xlabel='Max. original vertex degree', ylabel='Max. control rank', ylim=(-.3, 10.3))
    b.legend(loc='lower right', fontsize=7.4, handlelength=.85)

    checked = read('review_statistics.json')
    totals = [checked['cohorts'][k] for k in ('main144', 'new_confirmation72', 'sweep120', 'public24')]
    first = np.array([100*(r['ref']-r['none'])/r['ref'] for r in totals])
    second = np.array([100*(r['none']-r['out'])/r['ref'] for r in totals])
    y = np.arange(4)
    bars1 = c.barh(y, first, color=SPLIT_COLORS[0], height=.61, label=SPLIT_LABELS[0])
    bars2 = c.barh(y, second, left=first, color=SPLIT_COLORS[1], height=.61, label=SPLIT_LABELS[1])
    c.set(yticks=y, yticklabels=['Main', 'Confirmation', 'Sweeps/weights', 'Public graphs'],
          xlabel='Share of reference CX (%)', xlim=(0, 27.5))
    c.invert_yaxis()
    c.grid(axis='y', visible=False)
    for i, total in enumerate(first + second):
        c.annotate(f'{total:.2f}%', (total, i), xytext=(4, 0), textcoords='offset points',
                   va='center', fontsize=8.1)

    rows = checked['controlled']
    assert len(rows) == 72
    p_symbols = [(1, 'o', BLUE), (2, 's', TEAL), (4, '^', ORANGE)]
    for block, ax in [('pairs', d), ('redistribute', e)]:
        by_level = defaultdict(list)
        for r in rows:
            if r['block'] == block:
                by_level[r['coverage']].append(r)
        levels = sorted(by_level)
        x = np.arange(len(levels))
        def gain(r, route):
            return 100*(1-r['post'][route]/r['post']['matched_exact_clique'])
        before = [np.mean([gain(r, 'no_cache') for r in by_level[z]]) for z in levels]
        extra = [np.mean([gain(r, r['winner'])-gain(r, 'no_cache') for r in by_level[z]]) for z in levels]
        ax.bar(x, before, color=SPLIT_COLORS[0], width=.58)
        ax.bar(x, extra, bottom=before, color=SPLIT_COLORS[1], width=.58)
        for i, (p, marker, color) in enumerate(p_symbols):
            vals = [np.mean([gain(r, r['winner']) for r in by_level[z] if r['sweeps'] == p])
                    for z in levels]
            ax.scatter(x+(i-1)*.14, vals, marker=marker, s=18,
                       facecolors='white', edgecolors=color, linewidth=.85, zorder=3)
        ax.set(ylim=(0, 60), yticks=[0, 20, 40, 60], xlim=(-.48, len(x)-.52),
               xlabel='Nonsingleton coverage (%)')
        ax.set_xticks(x, [f'{100*z:.3f}'.rstrip('0').rstrip('.') for z in levels])
    d.set_ylabel('Share of matched exact CX (%)')
    e.tick_params(labelleft=False)
    symbols = [Line2D([], [], marker=m, color=color, markerfacecolor='white', ms=4,
                      ls='none', label=f'p={p}') for p, m, color in p_symbols]
    fig.legend(handles=[bars1, bars2, *symbols], loc='lower center', bbox_to_anchor=(.53, .014),
               ncol=5, fontsize=7.6, handlelength=1.2, handletextpad=.45, columnspacing=1.05)
    save(fig, '14_mechanism_structure', [EVAL, EDGES, STATS],
         {'panels': ['matched ablations, median and observed range', '62 labelled graphs',
                     'cohort totals relative to its envelope', 'pair coverage', 'fixed quotient redistribution'],
          'coverage_programs': 72, 'coverage_denominator': 'per-program matched exact reference CX',
          'aggregate_denominator': 'sum of the cohort fixed reference envelopes'})


def public_and_routing():
    fig = plt.figure(figsize=(WIDTH, 5.32))
    a = axes_inches(fig, LEFT, 3.13, PANEL_WIDTH, 1.86, 'a', 'Public-graph CX reductions')
    b = axes_inches(fig, RIGHT, 3.13, PANEL_WIDTH, 1.86, 'b', 'Net caching gains')
    c = axes_inches(fig, LEFT, .64, PANEL_WIDTH, 1.78, 'c', 'Paired CX after routing')
    d = axes_inches(fig, RIGHT, .64, PANEL_WIDTH, 1.78, 'd', 'Logical and physical reductions')
    groups = group(public, 'graph_id')
    names = sorted(groups, key=lambda k: (0 if k.startswith('myciel') else 1 if k.startswith('queen') else 2, k))
    assert len(names) == 12 and all(len(g) == 2 for g in groups.values())
    for i, name in enumerate(names):
        rows = sorted(groups[name], key=lambda r: r['order'] != 'ascending')
        gains = [saving(r) for r in rows]
        color = BLUE if name.startswith('myciel') else SLATE if name.startswith('queen') else TEAL
        a.plot(gains, [i-.12, i+.12], c=color, lw=.8, alpha=.6)
        a.scatter(gains[0], i-.12, s=21, marker='o', edgecolors=color, facecolors='white', lw=.85, zorder=3)
        a.scatter(gains[1], i+.12, s=19, marker='s', c=color, zorder=3)
        before = sum(r['all_candidate_metrics']['ours_none']['cx'] for r in rows)
        after = sum(r['ours_metrics']['cx'] for r in rows)
        gain = 100*(1-after/before)
        b.hlines(i, 0, gain, colors=color, lw=1.15)
        b.scatter(gain, i, c=color, s=18, zorder=3)
        b.text(.98, i, str(rows[0]['nontrivial_modules']), ha='right', va='center',
               transform=b.get_yaxis_transform(), fontsize=7.8, color=SLATE)
    for ax in (a, b):
        ax.set_ylim(11.6, -.6)
        ax.axvline(0, c=SLATE, lw=.7, ls='--')
        ax.set_yticks(range(12), [name.removesuffix('.col') for name in names])
        ax.tick_params(axis='y', labelsize=7.7)
    b.tick_params(axis='y', left=False, labelleft=False)
    a.set(xlim=(-1.6, 51), xlabel='Reduction vs. reference (%)')
    b.set(xlim=(-.35, 8.6), xticks=[0, 2, 4, 6], xlabel='Reduction vs. no cache (%)')
    b.text(.98, 1.018, 'Class count', transform=b.transAxes, ha='right', fontsize=7.7, color=SLATE)
    a.legend(handles=[Line2D([], [], c=SLATE, marker='o', mfc='white', ls='none', label='Ascending'),
                      Line2D([], [], c=SLATE, marker='s', ls='none', label='Fixed pseudorandom')],
             loc='center right', bbox_to_anchor=(.98, .53), fontsize=7.5, handlelength=.9)

    routes = evaluation['routing_final']
    by_topology = group(routes, 'topology')
    assert len(routes) == 132 and all(len(r) == 66 for r in by_topology.values())
    native = {r['case']: r for r in cohorts['main144']['programs']}
    joined = {r['case'] for r in routes} & native.keys()
    assert len(joined) == 16
    lo = min(r['ours_metrics']['cx'] for r in routes)*.8
    hi = max(r['reference_metric_envelope']['cx'] for r in routes)*1.15
    equality(c, lo, hi, log=True)
    limit = 5*np.ceil(max([saving(native[k]) for k in joined] +
                          [saving(r) for r in routes if r['case'] in joined])/5)
    equality(d, 0, limit)
    markers = []
    for top, name, color, marker in [('line', 'Linear', BLUE, 'o'), ('grid', 'Grid', TEAL, '^')]:
        rows = by_topology[top]
        markers.append(Line2D([], [], c=color, marker=marker, ls='none',
                              label=f'{name} {summed_saving(rows):.2f}%'))
        for source, filled in [('new_independent_matrix', True), ('fixed_existing_mechanism_graph', False)]:
            pts = [r for r in rows if r['routing_cohort'] == source]
            c.scatter([r['reference_metric_envelope']['cx'] for r in pts],
                      [r['ours_metrics']['cx'] for r in pts], s=16, marker=marker,
                      edgecolors=color, facecolors=color if filled else 'white', alpha=.75, linewidths=.65)
        pts = [r for r in rows if r['case'] in joined]
        assert len(pts) == 48
        d.scatter([saving(native[r['case']]) for r in pts], [saving(r) for r in pts],
                  marker=marker, c=color, s=14, alpha=.54, linewidths=0)
    c.set(xlabel='Physical reference CX', ylabel='Compiled CX with restoration')
    d.set(xlabel='Logical CX reduction (%)', ylabel='Physical CX reduction (%)')
    c.legend(handles=markers, loc='upper left', fontsize=7.6, handlelength=.9)
    d.text(.035, .96, '16 programs, 3 seeds per topology', transform=d.transAxes, va='top',
           fontsize=7.7, color=SLATE)
    save(fig, '15_public_routing', [EVAL],
         {'public_programs': 24, 'physical_records': 132, 'logical_physical_shared_programs': 16,
          'panels': ['public matched order', 'public cache-only increment', 'physical paired counts',
                     'paired logical and physical gain'], 'public_and_route_inputs_distinct': True})


def resources_and_scaling():
    fig = plt.figure(figsize=(WIDTH, 4.77))
    a = axes_inches(fig, LEFT, 2.75, 2.06, 1.72, 'a', 'Per-program gate trade-off')
    b = axes_inches(fig, 4.05, 2.75, 2.11, 1.72, 'b', 'Aggregate resource changes')
    c = axes_inches(fig, LEFT, .63, PANEL_WIDTH, 1.40, 'c', 'Wall-clock compilation time')
    d = axes_inches(fig, RIGHT, .63, PANEL_WIDTH, 1.40, 'd', 'Peak process RSS')

    for rows, title, color, marker in COHORTS:
        a.scatter([r['ours_metrics']['cx']/r['reference_CXwinner_metrics']['cx'] for r in rows],
                  [r['ours_metrics']['total_1q']/r['reference_CXwinner_metrics']['total_1q'] for r in rows],
                  s=14, c=color, marker=marker, alpha=.61, linewidths=0, label=title)
    a.axhline(1, c=SLATE, lw=.72, ls='--')
    a.axvline(1, c=SLATE, lw=.72, ls='--')
    a.set(xticks=[.7, .8, .9, 1], xlabel='CX ratio to paired reference', ylabel='1Q ratio to paired reference')
    a.legend(loc='upper left', fontsize=7.5, handlelength=.9)
    fields = ['cx', 'depth', 'total_1q', 'parameterized_rz']
    values = np.array([[100*(sum(r['ours_metrics'][m] for r in rows) /
                             sum(r['reference_CXwinner_metrics'][m] for r in rows)-1)
                       for m in fields] for rows in (main, confirm, sweep, public)])
    extreme = max(40, np.ceil(np.max(np.abs(values))/5)*5)
    b.grid(False)
    b.pcolormesh(np.arange(5)-.5, np.arange(5)-.5, values, cmap='RdBu_r',
                 norm=TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme),
                 edgecolors='white', linewidth=.35)
    b.invert_yaxis()
    b.set_xticks(range(4), ['CX', 'Depth', '1Q gates', 'Param. $R_z$'])
    b.set_yticks(range(4), ['Main', 'Confirmation', 'Sweeps/weights', 'Public graphs'])
    b.set_xlabel('Change in resources (%)')
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            b.text(j, i, f'{value:+.1f}', ha='center', va='center', fontsize=8.2,
                   color='white' if abs(value) > extreme*.62 else INK)

    summaries = read('linux_entry.json')['rows']
    trials = [json.loads(x['raw_text']) for x in read('linux_trials.json')['rows']]
    sizes = [r['n'] for r in summaries]
    assert len(trials) == 12 and all(r['status'] == 'passed' for r in trials)
    for ax, key, unit in [(c, 'seconds', 1), (d, 'peak_RSS_bytes', 2**20)]:
        ax.scatter([r['n'] for r in trials], [r[key]/unit for r in trials], s=23,
                   facecolors='white', edgecolors=ORANGE, linewidths=.85, zorder=3, label='Individual runs')
        ax.set_xscale('log', base=2)
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_xlabel('Data qubits')
    c.set_yscale('log')
    c.set_ylabel('Wall-clock time (s)')
    c.plot(sizes, [r['wall_median'] for r in summaries], c=TEAL, lw=1.1,
           marker='D', ms=3.2, zorder=4, label='Median')
    d.set_ylabel('RSS (MiB)')
    d.plot(sizes, [r['RSS_max_MiB'] for r in summaries], c=TEAL, lw=1.1,
           marker='D', ms=3.2, zorder=4, label='Maximum')
    for ax in (c, d):
        ax.legend(loc='upper left', fontsize=7.5, handlelength=1.05)
    save(fig, '16_resources_scaling', [EVAL, ENTRY, TRIALS],
         {'gate_resource_reference': 'same actual CX-winning reference circuit for each input',
          'gate_resource_totals': values.tolist(), 'linux_trials': 12,
          'linux_time': 'wall time median', 'linux_memory': 'maximum ru_maxrss across repetitions'})


if __name__ == '__main__':
    mechanism_and_structure()
    public_and_routing()
    resources_and_scaling()
