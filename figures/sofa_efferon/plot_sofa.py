"""Воспроизведение описательных графиков SOFA для Эфферон LPS."""

from pathlib import Path
import argparse
import csv
import json

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASE = Path(__file__).resolve().parent
TIME_HOURS = np.array([0, 24, 48, 72, 120])
FIELDS = ['record_id', 'survived_28d', 'discharged', *[f'sofa_{t}' for t in TIME_HOURS]]


def load_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Ожидаемые столбцы CSV: {", ".join(FIELDS)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Нет записей в таблице')
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Количество полей в строке не совпадает с заголовком CSV')
    ids = [row['record_id'].strip() for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError('record_id должен быть непустым и уникальным для каждой строки')
    for field in ('survived_28d', 'discharged'):
        if any(row[field] not in ('0', '1') for row in rows):
            raise ValueError(f'{field} должен быть 0 или 1')
    data = np.array([
        [float(row[f'sofa_{t}']) if row[f'sofa_{t}'].strip() else np.nan for t in TIME_HOURS]
        for row in rows
    ])
    for index, row in enumerate(rows):
        for column, time in enumerate(TIME_HOURS):
            if row[f'sofa_{time}'].strip():
                value = data[index, column]
                if not np.isfinite(value) or not 0 <= value <= 24 or value != int(value):
                    raise ValueError(f'{ids[index]}: SOFA {time} должен быть целым числом от 0 до 24')
    if not np.all(np.isfinite(data[:, 0])):
        raise ValueError('SOFA 0 должен быть указан для каждой записи')
    survived = np.array([row['survived_28d'] == '1' for row in rows])
    if not survived.any() or survived.all():
        raise ValueError('Для графика по исходам нужны оба 28-дневных исхода')
    return data, survived


def stats(data):
    """Квартили по доступным значениям, без заполнения пропусков."""
    quartiles = np.full((3, len(TIME_HOURS)), np.nan)
    observed = np.any(np.isfinite(data), axis=0)
    if observed.any():
        quartiles[:, observed] = np.nanquantile(
            data[:, observed], [.25, .5, .75], axis=0, method='linear'
        )
    return quartiles


def setup(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=16, pad=18)
    ax.set_xlim(-6, 126)
    ax.set_ylim(0, 24)
    ax.set_xticks(TIME_HOURS)
    ax.set_yticks(np.arange(0, 25, 4))
    ax.set_xlabel('Часы от первой гемосорбции', labelpad=12)
    ax.set_ylabel('SOFA, баллы', labelpad=8)
    ax.grid(axis='y', color='#E7EBF0', linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#A9B3BF')


def tracks(ax, data, color, alpha, style):
    for scores in data:
        ax.plot(
            TIME_HOURS, scores, color=color, alpha=alpha,
            lw=style['patient_line_width'], marker='o',
            ms=style['patient_marker_size'], zorder=2
        )
        if np.sum(np.isfinite(scores)) == 1:
            mask = np.isfinite(scores)
            ax.scatter(
                TIME_HOURS[mask], scores[mask], marker='D', s=55,
                facecolor='white', edgecolor=color, linewidth=1.8, zorder=6
            )


def summary(ax, data, color, style):
    quartiles = stats(data)
    ax.fill_between(
        TIME_HOURS, quartiles[0], quartiles[2], color=color,
        alpha=style['quartile_alpha'], zorder=1
    )
    ax.plot(
        TIME_HOURS, quartiles[1], color=color, lw=style['median_line_width'],
        marker='o', ms=style['median_marker_size'], zorder=5
    )


def save_figure(fig, name, output, style):
    fig.savefig(output / f'{name}.png', dpi=style['png_dpi'], transparent=True)
    fig.savefig(output / f'{name}.svg', transparent=True, metadata={'Date': None})
    plt.close(fig)


def new_figure(style):
    fig, ax = plt.subplots(figsize=style['figure_size'])
    fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.86)
    fig.text(.12, .965, 'Эфферон LPS', fontsize=12, va='top')
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description='Графики динамики SOFA при применении Эфферон LPS')
    parser.add_argument('--data', type=Path, default=BASE / 'data/sofa.csv')
    parser.add_argument('--style', type=Path, default=BASE / 'style.json')
    parser.add_argument('--output', type=Path, default=BASE / 'output')
    args = parser.parse_args()
    style = json.loads(args.style.read_text(encoding='utf-8'))
    data, survived = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': style['font_family'], 'font.size': style['font_size'],
        'svg.fonttype': 'none', 'svg.hashsalt': 'efferon-lps-sofa',
        'axes.labelcolor': '#263548', 'text.color': '#263548',
        'xtick.color': '#263548', 'ytick.color': '#263548'
    })
    blue = style['colors']['median']
    green = style['colors']['survivors']
    red = style['colors']['deceased']

    fig, ax = new_figure(style)
    setup(ax, 'А. Все индивидуальные траектории')
    tracks(ax, data, '#8C98A7', .48, style)
    summary(ax, data, blue, style)
    handles = [
        Line2D([0], [0], color=blue, lw=3, marker='o', label='Медиана'),
        Patch(facecolor=blue, alpha=style['quartile_alpha'], label='25–75-й процентили'),
        Line2D([0], [0], color='#8C98A7', lw=1.1, label='Отдельная запись')
    ]
    if np.any(np.sum(np.isfinite(data), axis=1) == 1):
        handles.append(Line2D(
            [0], [0], marker='D', mfc='white', mec='#8C98A7', ls='', label='Только SOFA 0'
        ))
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=11)
    save_figure(fig, 'SOFA_EFFERON_LPS_A', args.output, style)

    fig, ax = new_figure(style)
    setup(ax, 'Б. Траектории по 28-дневному исходу')
    for mask, color in [(~survived, red), (survived, green)]:
        tracks(ax, data[mask], color, .28, style)
        summary(ax, data[mask], color, style)
    ax.legend(handles=[
        Line2D([0], [0], color=green, lw=3, marker='o', label='Выжившие к 28-м суткам'),
        Line2D([0], [0], color=red, lw=3, marker='o', label='Умершие к 28-м суткам'),
        Patch(facecolor='#8595A8', alpha=.16, label='25–75-й процентили')
    ], loc='lower left', frameon=False, fontsize=11)
    save_figure(fig, 'SOFA_EFFERON_LPS_B', args.output, style)

    with (args.output / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['group', 'time_hours', 'n', 'q25', 'median', 'q75'])
        for label, scores in [('all', data), ('survivors', data[survived]), ('deceased', data[~survived])]:
            counts = np.sum(np.isfinite(scores), axis=0)
            quartiles = stats(scores)
            print(label, 'n=', counts.tolist(), 'quartiles=', quartiles.tolist())
            for column, time in enumerate(TIME_HOURS):
                values = ['' if not np.isfinite(value) else float(value) for value in quartiles[:, column]]
                writer.writerow([label, int(time), int(counts[column]), *values])


if __name__ == '__main__':
    main()
