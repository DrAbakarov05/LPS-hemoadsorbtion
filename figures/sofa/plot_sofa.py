"""Two standalone SOFA figures; transparent PNG, white PNG and editable SVG."""

from pathlib import Path
import argparse
import csv
import json
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def stats(a):
    """Pointwise quartiles over available observations, without imputation."""
    return np.nanquantile(a, [.25, .5, .75], axis=0, method='linear')


def build_figure(data, surv, x, style, by_outcome):
    ink, muted = '#253449', '#66758A'
    blue, green, red, gray = (style['colors'][key]
                              for key in ['median', 'survivors', 'deceased', 'patient'])
    fig, ax = plt.subplots(figsize=style['figure_size'])
    fig.subplots_adjust(left=.105, right=.965, bottom=.30, top=.76)
    fig.text(.105, .94, style['title'], fontsize=style['title_size'],
             fontweight='bold', color=ink, va='center')
    subtitle = (
        f'По 28-дневному исходу · {surv.sum()} выживших / {(~surv).sum()} умерших'
        if by_outcome else f'Все индивидуальные траектории · {len(data)} пациентов'
    )
    fig.text(.105, .888, subtitle, fontsize=16, color=muted, va='center')
    ax.set_xlim(-6, 126)
    ax.set_ylim(0, 24)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 25, 4))
    ax.set_xlabel('Часы от первой гемосорбции', labelpad=12)
    ax.set_ylabel('SOFA, баллы', labelpad=12)
    ax.grid(axis='y', color='#C5CED9', linewidth=.7, alpha=.55)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#9DAABA')
    ax.spines[['left', 'bottom']].set_linewidth(.8)
    ax.tick_params(length=0, pad=9)

    def tracks(a, color, alpha):
        for y in a:
            ax.plot(x, y, color=color, alpha=alpha,
                    lw=style['patient_line_width'], zorder=2)
            # A single observation has no segment: keep a short stroke, no dot.
            finite = np.isfinite(y)
            if finite.sum() == 1:
                ax.plot(x[finite], y[finite], marker='|', markersize=11,
                        markeredgewidth=2, color=color, linestyle='none', zorder=6)

    def summary(a, color):
        q = stats(a)
        ax.fill_between(x, q[0], q[2], color=color,
                        alpha=style['quartile_alpha'], linewidth=0, zorder=1)
        ax.plot(x, q[1], color=color, lw=style['median_line_width'], zorder=5)

    def counts(a, y, color, label):
        fig.text(.100, y, label, fontsize=12, color=color, va='center', ha='right')
        # Count positions follow the time ticks, including unequal spacing.
        for t, n in zip(x, np.sum(np.isfinite(a), axis=0)):
            fx = fig.transFigure.inverted().transform(ax.transData.transform((t, 0)))[0]
            fig.text(fx, y, str(n), fontsize=13, color=color,
                     ha='center', va='center')

    if by_outcome:
        for mask, color in [(~surv, red), (surv, green)]:
            tracks(data[mask], color, style['outcome_patient_alpha'])
            summary(data[mask], color)
        handles = [
            Line2D([0], [0], color=green, lw=3.4, label='Выжившие: медиана'),
            Line2D([0], [0], color=red, lw=3.4, label='Умершие: медиана'),
            Patch(facecolor=gray, alpha=.18, label='25–75-й процентили'),
        ]
        counts(data[surv], .155, green, 'n выживших')
        counts(data[~surv], .113, red, 'n умерших')
    else:
        tracks(data, gray, style['patient_alpha'])
        summary(data, blue)
        handles = [
            Line2D([0], [0], color=blue, lw=3.4, label='Медиана'),
            Patch(facecolor=blue, alpha=style['quartile_alpha'], label='25–75-й процентили'),
            Line2D([0], [0], color=gray, lw=1.3, label='Отдельный пациент'),
        ]
        counts(data, .155, muted, 'n измерений')
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.105, .823),
               ncol=3, frameon=False, fontsize=12, handlelength=2.3,
               columnspacing=1.7, borderaxespad=0)
    fig.text(.105, .053,
             'Тонкие линии — пациенты; штрих — единственное измерение SOFA.\n'
             'n — доступные измерения в каждой точке; пропуски не восстанавливались.',
             fontsize=11, color=muted, va='center', linespacing=1.6)
    return fig


def export(fig, out, stem, dpi):
    # Fixed canvas: 12 × 7.5 inches at 400 dpi gives exactly 4800 × 3000 px.
    # bbox_inches='tight' is omitted because it changes pixel dimensions.
    variants = [
        (f'{stem}_presentation.png', 'png', {'dpi': dpi, 'transparent': True}),
        (f'{stem}_publication.png', 'png', {'dpi': dpi, 'facecolor': 'white',
                                          'edgecolor': 'none', 'transparent': False}),
        (f'{stem}.svg', 'svg', {'transparent': True, 'metadata': {'Date': None}}),
    ]
    for name, fmt, options in variants:
        # Replace only a complete render, so previews never read a partial PNG.
        with BytesIO() as buffer:
            fig.savefig(buffer, format=fmt, **options)
            temporary = out / f'.{name}.tmp'
            temporary.write_bytes(buffer.getvalue())
            temporary.replace(out / name)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Два отдельных графика динамики SOFA')
    parser.add_argument('--data', type=Path, default=base / 'data/sofa.csv')
    parser.add_argument('--style', type=Path, default=base / 'style.json')
    parser.add_argument('--output', type=Path, default=base / 'output')
    args = parser.parse_args()
    style = json.loads(args.style.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    with args.data.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Нет пациентов в таблице')
    if any(r['survived_28d'] not in ('0', '1') for r in rows):
        raise ValueError('survived_28d должен быть 0 или 1')
    x = np.array([0, 24, 48, 72, 120])
    data = np.array([[float(r[f'sofa_{t}']) if r[f'sofa_{t}'].strip() else np.nan
                      for t in x] for r in rows])
    surv = np.array([r['survived_28d'] == '1' for r in rows])
    if not np.all((data[np.isfinite(data)] >= 0) & (data[np.isfinite(data)] <= 24)):
        raise ValueError('SOFA должен быть в диапазоне 0–24')
    if not np.all(np.isfinite(data[:, 0])):
        raise ValueError('SOFA 0 должен быть указан для каждого пациента')
    if not surv.any() or surv.all():
        raise ValueError('Для графика по исходу нужны оба 28-дневных исхода')
    plt.rcParams.update({
        'font.family': style['font_family'], 'font.size': style['font_size'],
        'svg.fonttype': 'none', 'svg.hashsalt': 'lps-gema-sofa',
        'axes.labelcolor': '#253449', 'text.color': '#253449',
        'xtick.color': '#253449', 'ytick.color': '#253449',
        'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    })
    for by_outcome, stem in [(False, 'SOFA_LPS_GEMA_all'),
                             (True, 'SOFA_LPS_GEMA_outcomes')]:
        fig = build_figure(data, surv, x, style, by_outcome)
        export(fig, args.output, stem, style['png_dpi'])
        plt.close(fig)
    for label, a in [('all', data), ('survivors', data[surv]), ('deceased', data[~surv])]:
        print(label, 'n=', np.sum(np.isfinite(a), axis=0).tolist(), 'quartiles=', stats(a).tolist())


if __name__ == '__main__':
    main()
