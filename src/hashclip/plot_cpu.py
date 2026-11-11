"""Exploratory plots from measured CPU rows, without invented summary values."""

from pathlib import Path


def plot_systems(rows, selected, output: Path, dataset: str, *, stage="Test"):
    """Two empirical Pareto panels per direction/batch; frozen validation points.

    Seed means use only shared M values. Stars average independently selected
    seed operating points and are hollow when any seed misses the test target.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .final_cpu import frontier

    output.mkdir(parents=True, exist_ok=True)
    styles = {
        "flat_ip": ("Exact (FlatIP)", "#555555", "s", "-"),
        "hashclip": ("HashCLIP · Flickr weights", "#C65D26", "o", "-"),
        "hashclip_coco": ("HashCLIP · COCO weights", "#C65D26", "D", "--"),
        "lsh": ("LSH + refinement", "#8E79B2", "v", ":"),
        "ivf_flat": ("IVF-Flat", "#3775A9", "^", "-"),
        "hnsw": ("HNSW", "#31846E", "P", "-"),
        "ivf_pq": ("IVF-PQ + refinement", "#B69035", "X", "-.")}
    methods = [m for m in styles if any(r['method']==m for r in rows)]
    for direction in ('t2i','i2t'):
        for batch in sorted({r['batch'] for r in rows}):
            subset = [r for r in rows if r['direction']==direction and r['batch']==batch]
            if not subset:
                continue
            with plt.rc_context({'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none'}):
                fig, axes = plt.subplots(1,2,figsize=(10.2,4.7),sharey=True)
                for method in methods:
                    raw = [r for r in subset if r['method']==method]
                    if not raw:
                        continue
                    points = raw
                    if method in ('hashclip','hashclip_coco','lsh'):
                        seeds = {r['seed'] for r in raw}
                        counts = sorted({r['M'] for r in raw})
                        points = []
                        for count in counts:
                            group = [r for r in raw if r['M']==count]
                            if {r['seed'] for r in group} != seeds:
                                continue
                            points.append({key:float(np.mean([r[key] for r in group])) for key in
                                           ('final_recall_at_10','final_complete_at_10','amortized_p50_ms')})
                    label, color, marker, line = styles[method]
                    for ax, metric in zip(axes,('final_recall_at_10','final_complete_at_10')):
                        points_front = sorted(frontier(points,metric),key=lambda r:r[metric])
                        ax.scatter([100*r[metric] for r in points],[r['amortized_p50_ms'] for r in points],
                                   color=color,marker=marker,s=15,alpha=.22,zorder=1)
                        ax.plot([100*r[metric] for r in points_front],[r['amortized_p50_ms'] for r in points_front],
                                color=color,marker=marker,markersize=4,ls=line,label=label,lw=1.5)
                        for target in (.95,.99):
                            picks=[r for r in selected if r['method']==method and r['direction']==direction
                                   and r['batch']==batch and r['target_mean_recall_at_10']==target
                                   and r.get('target_met_on_validation')]
                            if picks:
                                attained=all(r.get('target_met_on_test',False) for r in picks)
                                ax.scatter([100*np.mean([r[metric] for r in picks])],
                                           [np.mean([r['amortized_p50_ms'] for r in picks])],
                                           marker='*',s=110,edgecolors=color,facecolors=color if attained else 'white',zorder=5)
                for ax, title in zip(axes,('Mean top-10 retention (%)','Complete top-10 recovery (%)')):
                    ax.set_xlabel(title)
                    ax.set_yscale('log')
                    ax.set_xlim(0,101)
                    ax.grid(alpha=.18)
                axes[0].set_ylabel('Query p50 (ms)' if batch==1 else 'Amortized p50 (ms/query)')
                handles,labels=axes[0].get_legend_handles_labels()
                fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.92),frameon=False,fontsize=8.7)
                fig.suptitle(f"{dataset.replace('_',' ')} · {direction.upper().replace('2',' → ')} · batch {batch} · {stage}",y=.99)
                domain_note = ('\nExpanded gallery contains COCO training images' if dataset == 'coco_large' and 'hashclip_coco' in methods
                               else '\nExpanded gallery contains Flickr training images' if dataset == 'flickr_large' else '')
                fig.text(.5,.015,'Frozen validation configurations · binary curves: mean across seeds 0, 1, 42\n'
                         'Stars: validation-selected 95% / 99% points; hollow: at least one test seed misses target' + domain_note,
                         ha='center',fontsize=8.3)
                fig.tight_layout(rect=(0,.12 if domain_note else .09,1,.80))
                for suffix in ('pdf','png'):
                    fig.savefig(output/f'{direction}_b{batch}.{suffix}',dpi=180)
                plt.close(fig)


def plot_representations(rows: list[dict], output: Path, title: str,
                         *, methods=("hashclip", "lsh", "itq", "pq", "opq"), footnote=None):
    """Full-scan, equal-byte test curves; only budgets common to every seed/method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    output.mkdir(parents=True, exist_ok=True)
    styles = {
        "hashclip": ("HashCLIP-128", "#b64f29", "-", "o"),
        "lsh": ("LSH-128", "#417a9b", "--", "s"),
        "itq": ("ITQ-128", "#6a6097", "-.", "^"),
        "pq": ("PQ16×8", "#777777", ":", "v"),
        "opq": ("OPQ16", "#498556", "-", "D"),
        "hashclip_coco": ("HashCLIP (COCO)", "#C65D26", "--", "s"),
    }
    if "hashclip_coco" in methods:
        _, color, line, marker = styles["hashclip"]
        styles["hashclip"] = ("HashCLIP (Flickr)", color, line, marker)
    styles = {method: styles[method] for method in methods}
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")
        for col, direction in enumerate(("t2i", "i2t")):
            values = [r for r in rows if r["split"] == "test" and r["direction"] == direction]
            groups = {(method, seed): {int(r["M"]): r for r in values
                                      if r["method"] == method and int(r["seed"]) == seed}
                      for method in styles for seed in (0, 1, 42)}
            budgets = sorted(set.intersection(*(set(group) for group in groups.values())))
            if not budgets:
                raise ValueError("No shared candidate budget across all methods and seeds")
            size = int(values[0]["gallery_size"])
            for row, metric in enumerate(("candidate_recall_at_10", "candidate_complete_at_10")):
                ax = axes[row, col]
                for method, (label, color, line, marker) in styles.items():
                    scores = 100 * np.array([[float(groups[method, seed][budget][metric])
                                              for budget in budgets] for seed in (0, 1, 42)])
                    ax.plot(budgets, scores.mean(axis=0), color=color, linestyle=line, marker=marker,
                            markersize=3.5, linewidth=1.6, label=label)
                    ax.fill_between(budgets, scores.min(axis=0), scores.max(axis=0), color=color, alpha=.12)
                ax.set(xscale="log", xlim=(budgets[0], budgets[-1]), ylim=(-2, 102),
                       xlabel="Candidate budget M", ylabel="Mean top-10 retention (%)" if row == 0
                       else "Complete top-10 recovery (%)", title=f"{'Text → image' if col == 0 else 'Image → text'} · {size:,} gallery items")
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
                ax.grid(alpha=.18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(title, y=.985, fontsize=13)
        fig.legend(handles, labels, ncol=3 if len(styles)>5 else len(styles), loc="upper center", bbox_to_anchor=(.5, .955), frameon=False)
        adc = "PQ/OPQ use ADC · " if "pq" in styles or "opq" in styles else ""
        fig.text(.5, .018, "All test queries · batch 32 · mean and min–max across seeds 0, 1, 42\n"
                 f"16 bytes/item · full scans · {adc}shared continuous refinement" + (f"\n{footnote}" if footnote else ""),
                 ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, .11 if footnote else .085, 1, .9))
        fig.savefig(output / "candidate_curves.png", dpi=180)
        fig.savefig(output / "candidate_curves.pdf")
        plt.close(fig)


def plot_checkpoint_selection(rows: list[dict], output: Path):
    """Paired selectors on the same measured training trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    colors = {0: "#b64f29", 1: "#417a9b", 42: "#6a6097"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for seed, color in colors.items():
        values = sorted([r for r in rows if int(r["seed"]) == seed], key=lambda r: int(r["epoch"]))
        for ax, field in zip(axes, ("semantic_r1_mean", "candidate_selection_percent")):
            ax.plot([int(r["epoch"]) for r in values], [float(r[field]) for r in values],
                    ".-", color=color, linewidth=1.6, label=f"Seed {seed}")
            for row in values:
                if str(row["selected_by_r1"]).lower() == "true":
                    ax.scatter(int(row["epoch"]), float(row[field]), marker="^", s=90,
                               color=color, edgecolor="white", linewidth=.8, zorder=4)
                if str(row["selected_by_candidate"]).lower() == "true":
                    ax.scatter(int(row["epoch"]), float(row[field]), marker="*", s=140,
                               color=color, edgecolor="white", linewidth=.6, zorder=5)
    for ax, title in zip(axes, ("Mean semantic Recall@1 (M=1,000)", "Reference top-10 candidate retention")):
        ax.set(title=title, xlabel="Training epoch", ylabel="Validation score (%)", ylim=(65, 85),
               xticks=(1, 5, 10, 15), xlim=(.5, 15.5))
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], marker="^", color="none", markerfacecolor="#444444", markersize=8),
                Line2D([], [], marker="*", color="none", markerfacecolor="#444444", markersize=10)]
    labels += ["Recall selection", "Candidate selection"]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.text(.5, .015, "Flickr30k validation. Candidate score averages both directions at 1%, 2%, 5%, 10% of the gallery.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .9))
    fig.savefig(output / "checkpoint_selection.png", dpi=180)
    fig.savefig(output / "checkpoint_selection.pdf")
    plt.close(fig)


def plot_results(rows: list[dict], output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    colors = {"hashclip": "#b64f29", "lsh": "#417a9b", "ivf_flat": "#6a6097", "flat_ip": "#498556"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for col, direction in enumerate(("t2i", "i2t")):
        for row, metric in enumerate(("candidate_recall_at_10", "candidate_complete_at_10")):
            ax = axes[row, col]
            for method in ("hashclip", "lsh"):
                subset = [r for r in rows if r["direction"] == direction and r["batch"] == 1 and r["method"] == method]
                for j, seed in enumerate(sorted({r["seed"] for r in subset})):
                    points = sorted([r for r in subset if r["seed"] == seed], key=lambda r: r["M"])
                    ax.plot([r["effective_M"] for r in points], [r[metric] for r in points],
                            ".-", color=colors[method], alpha=0.7, label=method if j == 0 else None)
            ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel="Candidate budget (capped at gallery size)",
                   ylabel="Mean top-10 retention" if row == 0 else "Complete top-10 retention rate",
                   title=f"{direction.upper()} · batch 1")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "candidate_retention.png", dpi=180)
    fig.savefig(output / "candidate_retention.pdf")
    plt.close(fig)

    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(len(batches), 2, figsize=(10, 3.6 * len(batches)), squeeze=False)
    for row, batch in enumerate(batches):
        for col, direction in enumerate(("t2i", "i2t")):
            ax = axes[row, col]
            for method, color in colors.items():
                points = [r for r in rows if r["direction"] == direction and r["batch"] == batch and r["method"] == method]
                ax.scatter([r["amortized_p50_ms"] for r in points], [r["final_recall_at_10"] for r in points],
                           s=20, color=color, alpha=0.7, label=method)
            ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel="Median latency / query (ms; amortized for batches)",
                   ylabel="Mean reference top-10 overlap", title=f"{direction.upper()} · batch {batch}")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    names = {"hashclip": "HashCLIP", "lsh": "LSH + rerank", "ivf_flat": "IVF-Flat", "flat_ip": "Exact (FlatIP)"}
    fig.legend(handles, [names[label] for label in labels], loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "quality_latency.png", dpi=180)
    fig.savefig(output / "quality_latency.pdf")
    plt.close(fig)



def plot_scaling(rows: list[dict], output: Path):
    """Validation-selected latency across gallery sizes; retain missed targets."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    colors = {"hashclip": "#b64f29", "lsh": "#417a9b", "ivf_flat": "#6a6097", "flat_ip": "#498556"}
    names = {"hashclip": "HashCLIP", "lsh": "LSH + rerank", "ivf_flat": "IVF-Flat", "flat_ip": "Exact (FlatIP)"}
    output.mkdir(parents=True, exist_ok=True)
    for target in sorted({float(r["target_mean_recall_at_10"]) for r in rows}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        for row, batch in enumerate((1, 32)):
            for col, direction in enumerate(("t2i", "i2t")):
                ax = axes[row, col]
                points = [r for r in rows if r["direction"] == direction and int(r["batch"]) == batch
                          and float(r["target_mean_recall_at_10"]) == target and r.get("gallery_size")]
                sizes = sorted({int(r["gallery_size"]) for r in points})
                for method, color in colors.items():
                    x, y, lo, hi, missed = [], [], [], [], []
                    for n in sizes:
                        group = [r for r in points if r["method"] == method and int(r["gallery_size"]) == n]
                        if not group:
                            continue
                        values = [float(r["amortized_p50_ms"]) for r in group]
                        x.append(n)
                        y.append(float(np.mean(values)))
                        lo.append(min(values))
                        hi.append(max(values))
                        missed.append(any(str(r["target_met_on_test"]).lower() != "true" for r in group))
                    if not x:
                        continue
                    ax.errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - y], fmt="o-", color=color,
                                linewidth=1.6, markersize=4, capsize=3, label=names[method])
                    for n, value, is_missed in zip(x, y, missed):
                        if is_missed:
                            ax.plot(n, value, "x", color="black", markersize=8, markeredgewidth=1.5)
                ax.set(xscale="log", yscale="log", xlabel="Gallery items",
                       ylabel="Median latency / query (ms)",
                       title=f"{direction.upper()} · {'single query' if batch == 1 else 'batch 32 (amortized)'}")
                ax.xaxis.set_major_locator(FixedLocator(sizes))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
                ax.minorticks_off()
                ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f"Validation-selected settings · {target:.0%} mean top-10 retention target", fontsize=12, y=.99)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=4)
        fig.text(.5, .012, "Points: mean of seed medians; bars: seed range. ×: at least one seed misses the test target.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .9))
        name = f"scaling_{round(target * 100)}"
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)


def plot_ivf_validation(rows: list[dict], output: Path):
    """Show the measured validation search-budget curves, separately by nlist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    if any(r["split"] != "val" for r in rows):
        raise ValueError("IVF tuning plots must use validation rows")
    points = [r for r in rows if r["method"] == "ivf_flat"]
    lists = sorted({int(r["nlist"]) for r in points})
    colors = {nl: plt.get_cmap("tab10")(i) for i, nl in enumerate(lists)}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    legend = {}
    for row, batch in enumerate((1, 32)):
        for col, direction in enumerate(("t2i", "i2t")):
            ax = axes[row, col]
            for nl in lists:
                group = sorted([r for r in points if int(r["batch"]) == batch
                                and r["direction"] == direction and int(r["nlist"]) == nl],
                               key=lambda r: int(r["nprobe"]))
                if not group:
                    continue
                line, = ax.plot([float(r["amortized_p50_ms"]) for r in group],
                                [float(r["final_recall_at_10"]) for r in group],
                                ".-", color=colors[nl], linewidth=1.1, markersize=4)
                legend[f"nlist={nl}"] = line
            exact = [r for r in rows if r["method"] == "flat_ip" and r["direction"] == direction
                     and int(r["batch"]) == batch]
            if len(exact) != 1:
                raise ValueError("Expected one current FlatIP measurement per panel")
            line, = ax.plot(float(exact[0]["amortized_p50_ms"]),
                            float(exact[0]["final_recall_at_10"]), "k*", markersize=9)
            legend["FlatIP"] = line
            for target in (.95, .99):
                ax.axhline(target, color="grey", linestyle=":", linewidth=.8)
            ax.set(xscale="log", ylim=(.90, 1.003), xlabel="Median latency / query (ms)",
                   ylabel="Mean reference top-10 overlap",
                   title=f"{direction.upper()} · {'single query' if batch == 1 else 'batch 32 (amortized)'}")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.15)
    labels = [f"nlist={nl}" for nl in lists] + ["FlatIP"]
    fig.suptitle("IVF validation tuning · high-retention region", y=.99, fontsize=12)
    fig.legend([legend[label] for label in labels], labels, loc="upper center",
               bbox_to_anchor=(.5, .955), ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .88))
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"ivf_validation.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Plot measured CPU retrieval results")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scaling", nargs="+", type=Path, help="selected_test.csv files, one per gallery scale")
    mode.add_argument("--ivf-validation", type=Path, help="val.csv from the IVF refinement")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.scaling or [args.ivf_validation]:
        with path.open() as f:
            records.extend(csv.DictReader(f))
    if args.scaling:
        plot_scaling(records, args.output)
    else:
        plot_ivf_validation(records, args.output)
