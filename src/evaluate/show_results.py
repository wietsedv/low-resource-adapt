from argparse import ArgumentParser
from pathlib import Path
import json
import yaml
import os

import numpy as np
from tabulate import tabulate


def create_table(data, data_names, args):
    data_names = sorted(data_names)

    headers = ['model', 'pos-lang', 'pos', 'pos-ckpt', 'mlm', 'mlm-ckpt']
    table = []
    for id_vals in sorted(data):
        # table.append(
        #     list(id_vals) + [
        #         data[id_vals][n][0] /
        #         data[id_vals][n][1] if n in data[id_vals] else 0
        #         for n in data_names
        #     ])
        table.append(
            list(id_vals) + [
                data[id_vals][n] if n in data[id_vals] else [0]
                for n in data_names
            ])

    nhead = len(headers)
    filtered_table = [[
        ''
        if i > 0 and table[i][:j + 1] == table[i - 1][:j + 1] else table[i][j]
        for j in range(nhead)
    ] + [np.mean(acc) for acc in table[i][nhead:]] for i in range(len(table))]

    headers = headers + data_names
    print(tabulate(filtered_table, headers=headers, floatfmt='.3f'))

    return table, headers


def plot_table(table, headers, path, xlabel, multi=False):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    assert headers[6:] == ['de', 'en', 'fy', 'gro', 'nl']

    data_names = ['gum', 'partut', 'gsd', 'hdt', 'alpino', 'lassysmall']
    table.sort(key=lambda row: data_names.index(row[2]))

    # set height of bar
    bars_en = [[np.mean(row[7][i]) for row in table] for i in range(2)]
    bars_de = [[np.mean(row[6][i]) for row in table] for i in range(2)]
    bars_nl = [[np.mean(row[10][i]) for row in table] for i in range(2)]
    bars_fy = [[np.mean(row[8][i]) for row in table] for i in range(2)]
    bars_gro = [[np.mean(row[9][i]) for row in table] for i in range(2)]

    err_en = [[np.abs(np.diff(row[7][i]))[0] / 2 for row in table]
              for i in range(2)]
    err_de = [[np.abs(np.diff(row[6][i]))[0] / 2 for row in table]
              for i in range(2)]
    err_nl = [[np.abs(np.diff(row[10][i]))[0] / 2 for row in table]
              for i in range(2)]
    # err_fy = [[np.abs(np.diff(row[8][i])) for row in table] for i in range(2)]
    # err_gro = [[np.abs(np.diff(row[9][i])) for row in table] for i in range(2)]

    # Set position of bar on X axis
    # if multi:
    #     barWidth = 1 / 6
    #     r = np.arange(len(bars_en[0]))
    #     r_en = [x - barWidth for x in r]
    #     r_de = [x + barWidth for x in r_en]
    #     r_nl = [x + barWidth for x in r_de]
    #     r_fy = [x + barWidth for x in r_nl]
    #     r_gro = [x + barWidth for x in r_fy]
    # else:
    bars_en = [[vals[i] if i < 2 else 0 for i in range(6)] for vals in bars_en]
    bars_de = [[vals[i] if i >= 2 and i < 4 else 0 for i in range(6)]
               for vals in bars_de]
    bars_nl = [[vals[i] if i >= 4 else 0 for i in range(6)]
               for vals in bars_nl]

    err_en = [[e[i] if v[i] > 0 else 0 for i in range(6)]
              for v, e in zip(bars_en, err_en)]
    err_de = [[e[i] if v[i] > 0 else 0 for i in range(6)]
              for v, e in zip(bars_de, err_de)]
    err_nl = [[e[i] if v[i] > 0 else 0 for i in range(6)]
              for v, e in zip(bars_nl, err_nl)]

    barWidth = 1 / 4
    r = np.arange(len(bars_en[0]))
    r_en = [x - barWidth for x in r]
    r_de = r_en
    r_nl = r_en
    r_fy = [x + barWidth for x in r_nl]
    r_gro = [x + barWidth for x in r_fy]

    for y in range(1, 10):
        plt.axhline(y / 10,
                    linestyle='--',
                    linewidth=1,
                    color='#cccccc',
                    zorder=0)

    colors = cm.get_cmap('tab20')
    patterns = ('--', 'xx', '\\\\', 'oo', '..')
    edgecolor = (0, 0, 0, 1)

    # Make the plot
    for i in range(1, 2):
        print(r_en, bars_en[i], err_en[i])
        plt.bar(r_en,
                bars_en[i],
                yerr=err_en[i],
                capsize=4,
                color=colors(0 + i),
                hatch=patterns[0],
                width=barWidth,
                edgecolor=edgecolor,
                label='English' if i == 1 else None)
        plt.bar(r_de,
                bars_de[i],
                yerr=err_de[i],
                capsize=4,
                color=colors(12 + i),
                hatch=patterns[1],
                width=barWidth,
                edgecolor=edgecolor,
                label='German' if i == 1 else None)
        plt.bar(r_nl,
                bars_nl[i],
                yerr=err_nl[i],
                capsize=4,
                color=colors(2 + i),
                hatch=patterns[2],
                width=barWidth,
                edgecolor=edgecolor,
                label='Dutch' if i == 1 else None)

    for i in range(2):
        plt.bar(r_fy,
                bars_fy[i],
                color=colors(6 + i),
                hatch=patterns[3],
                width=barWidth,
                edgecolor=edgecolor,
                label='West Frisian' if i == 0 else None)
        plt.bar(r_gro,
                bars_gro[i],
                color=colors(4 + i),
                hatch=patterns[4],
                width=barWidth,
                edgecolor=edgecolor,
                label='Gronings' if i == 0 else None)

    if not multi:
        plt.axvline(1.5, linestyle='--', color='#aaaaaa')
        plt.axvline(3.5, linestyle='--', color='#aaaaaa')

    # Add xticks on the middle of the group bars
    # plt.title(title)

    # ticks = [r + barWidth if multi else r for r in range(len(r))]
    ticks = list(range(len(r)))

    plt.ylim((0, 1))
    plt.ylabel('Accuracy')
    plt.xlabel(xlabel)
    plt.xticks(
        ticks,
        ['en-GUM', 'en-ParTUT', 'de-GSD', 'de-HDT', 'nl-Alpino', 'nl-Lassy'])

    # plt.axes(0).set_axisbelow(True)
    # plt.yaxis.grid(color='gray', linestyle='dashed')

    # Create legend & Show graphic
    fig = plt.gcf()
    fig.set_size_inches(5, 3.5)
    plt.legend(loc='lower right')
    plt.tight_layout(pad=0)
    plt.savefig(path)
    plt.close()

    print(f'plot saved to {path}')


def make_plots(headers, table, orig_table, args):
    out_dir = Path('data/plots')
    os.makedirs(out_dir, exist_ok=True)

    mlm_langs = {row[4] for row in table}
    assert len(mlm_langs) == 1

    for multi in [False, True]:
        table_data = [[
            (table[i][j], orig_table[i][j]) if j > 5 else table[i][j]
            for j in range(len(table[i]))
        ] for i in range(len(table)) if (table[i][0] == 'mbert') == multi]

        if len(table_data) == 0:
            print('WARNING: skipping', 'multi' if multi else 'mono')
            continue

        print('')
        print(tabulate(table_data))

        plot_name = f'{args.name}_{args.config}_{"multi" if multi else "mono"}.{"pdf" if args.latex else "png"}'
        plot_table(
            table_data,
            headers,
            out_dir / plot_name,
            xlabel='Training data',
            # xlabel=f'M{"ulti" if multi else "ono"}lingual BERT',
            multi=multi)


def plot_data_curve(results, path):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    # import seaborn as sns

    langs = ['en', 'de', 'nl']
    lang_colors = {'en': 0, 'nl': 2, 'de': 12}
    data = {lang: {} for lang in langs}

    for src_lang, model, dataset, subset, score in results:
        # name = (model, dataset)
        if model not in data[src_lang]:
            data[src_lang][model] = {}

        if subset not in data[src_lang][model]:
            data[src_lang][model][subset] = []
        data[src_lang][model][subset].append(score)

    colors = cm.get_cmap('tab20')

    for lang in langs:
        for j, (model, subsets) in enumerate(data[lang].items()):
            x = sorted(subsets)
            y = [np.mean(data[lang][model][s]) for s in x]
            plt.plot(x,
                     y,
                     label=f'{lang}-{model}',
                     color=colors(lang_colors[lang] + j))

    plt.ylim((0, 1))
    plt.ylabel('Accuracy')
    plt.xlabel('Subset size')
    plt.legend(title='Model', loc='lower right')
    plt.tight_layout(pad=0)
    plt.savefig(path)
    plt.close()
    print(f'plot saved to {path}')


def make_data_plots(headers, table, args):
    out_dir = Path('data/plots')
    os.makedirs(out_dir, exist_ok=True)

    mlm_langs = sorted({row[4].split('-')[0] for row in table})
    print(mlm_langs)

    for tgt_lang in mlm_langs:
        lang_table = [row for row in table if row[4].split('-')[0] == tgt_lang]

        if tgt_lang == 'frisian':
            score_idx = headers.index('fy')
        elif tgt_lang == 'groningen':
            score_idx = headers.index('gro')
        else:
            print(f'cannot plot target language {tgt_lang}')
            exit(1)

        print(f'\n{tgt_lang}')
        results = []
        for row in lang_table:
            model, src_lang, dataset = row[:3]
            subset = int(row[4].split('-')[-1])
            score = row[score_idx]
            results.append((src_lang, model, dataset, subset, score))

        results = sorted(results)
        plot_name = f'{args.name}_{args.config}_data.{"pdf" if args.latex else "png"}'
        plot_data_curve(results, out_dir / plot_name)


def select_best_mlm(table, headers):
    for i in range(6, len(headers)):
        print(f'\nData: {headers[i]}')
        prev_id = None
        best_ckpt = None
        best_score = 0
        prev_ckpt = 0

        out_data = {}
        for row in table:
            id_ = tuple(row[:5])
            if prev_id != id_:
                if prev_id is not None:
                    data_name = tuple(prev_id[:-1])
                    if data_name not in out_data:
                        out_data[data_name] = []
                    out_data[data_name].append(
                        (prev_id[-1], best_ckpt, prev_ckpt, best_score))

                prev_id, best_ckpt, best_score = id_, None, 0
            if row[i] > best_score:
                best_ckpt = row[5]
                best_score = row[i]
            prev_ckpt = row[5]

        data_name = tuple(prev_id[:-1])
        if data_name not in out_data:
            out_data[data_name] = []
        out_data[data_name].append(
            (prev_id[-1], best_ckpt, prev_ckpt, best_score))

        for (model, lang, data, pos_ckpt), bests in out_data.items():
            print(f'{model}_pos_{lang}_{data}:')
            print(f'  - {{pos: {pos_ckpt:_},')
            for name, ckpt, last_ckpt, score in bests:
                print(
                    f'     {name}: {ckpt:_},  # Dev: {score:.3f}, Last: {last_ckpt:_}'
                )

            print('  }')


def load_results(name, lang, args):
    pos_version, mlm_version = name.split('_')

    data = {}
    data_names = set()

    cfg_path = f'config-{args.config}.yml' if args.config is not None else 'config.yml'
    with open(cfg_path) as f:
        cfg = yaml.full_load(f)

    orig_data_names = cfg['data']['test' if args.test else 'dev']
    # src_data_names = set()
    if type(orig_data_names) == dict:
        # src_data_names = orig_data_names['source']
        orig_data_names = orig_data_names['source'] + orig_data_names['target']

    if not args.all_data:
        if args.ungroup:
            data_names = set(orig_data_names)
        else:
            data_names = {n.split('_')[0] for n in orig_data_names}

    pos_checkpoints = cfg['checkpoints']['pos'][pos_version]
    mlm_checkpoints = cfg['checkpoints']['mlm'][
        mlm_version] if mlm_version != 'None' else None

    base_dir = Path('data') / 'results' / name
    for path in base_dir.glob('*/*/*.json'):
        data_name = path.name.split('.')[0]
        # data_lang = path.name.split('.')[0].split('_')[0]
        model_name = path.parent.parent.name.split('_')[0]

        mlm_parts = path.parent.name.split('-')
        mlm_full_name, mlm_ckpt = mlm_parts[0], mlm_parts[-1]
        if len(mlm_parts) > 2:
            mlm_full_name += '-' + mlm_parts[1]
        mlm_name = mlm_full_name.split('_')[-1]
        mlm_ckpt = int(mlm_ckpt)

        pos_full_name, pos_ckpt = path.parent.parent.name.split('-')
        pos_name = pos_full_name.split('_')[-1]
        pos_ckpt = int(pos_ckpt)

        pos_lang = path.parent.parent.name.split('_')[-2]

        if mlm_name == 'none' and lang is not None:
            mlm_name = lang

        if args.model is not None and model_name != args.model:
            continue

        if args.data is not None and data_name != args.data:
            continue

        if lang is not None and mlm_name.split('-')[0] != lang:
            continue

        if args.pos_ckpt is not None and pos_ckpt != args.pos_ckpt:
            continue

        if args.mlm_ckpt is not None and mlm_ckpt != args.mlm_ckpt:
            continue

        if not args.all_data and data_name not in orig_data_names:
            continue

        if not args.all:
            if mlm_full_name != 'none' and (
                    mlm_full_name not in mlm_checkpoints
                    or mlm_ckpt not in mlm_checkpoints[mlm_full_name]):
                continue

            # if not args.subsets:
            if pos_full_name not in pos_checkpoints:
                continue
            # if len(pos_checkpoints[pos_full_name]) > 1:
            valid = False
            for ckpt in pos_checkpoints[pos_full_name]:
                best = False
                if type(ckpt) == dict:
                    if mlm_name in ckpt and ckpt[
                            mlm_name] != mlm_ckpt and mlm_ckpt > 0:
                        # print(mlm_name, ckpt[mlm_name], mlm_ckpt)
                        continue
                    best = mlm_name in ckpt['best'] if 'best' in ckpt else False
                    ckpt = ckpt['pos']
                if args.best and not best and len(
                        pos_checkpoints[pos_full_name]) > 1:
                    continue
                if ckpt == pos_ckpt:
                    valid = True
                    break
            if not valid:
                continue

        if not args.ungroup:
            data_name = data_name.split('_')[0]

        data_names.add(data_name)

        with open(path) as f:
            r = json.load(f)
            # sup = r['macro avg']['support']
            # cor = round(r['accuracy'] * sup)
            acc = r['accuracy']

        id_vals = (model_name, pos_lang, pos_name, pos_ckpt,
                   mlm_name if mlm_ckpt > 0 else f'original_{mlm_name}',
                   mlm_ckpt)
        if id_vals not in data:
            data[id_vals] = {}
        if data_name in data[id_vals]:
            # pcor, psupp = data[id_vals][data_name]
            # cor, sup = cor + pcor, sup + psupp
            data[id_vals][data_name].append(acc)
        else:
            data[id_vals][data_name] = [acc]

        # data[id_vals][data_name] = (cor, sup)

    return data, data_names


def main():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('-d', '--data', default=None)
    parser.add_argument('-l', '--lang', default=None)
    parser.add_argument('-pc', '--pos-ckpt', default=None, type=int)
    parser.add_argument('-mc', '--mlm-ckpt', default=None, type=int)
    parser.add_argument('--ungroup', action='store_true')
    # parser.add_argument('--min', type=float, default=0.)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--all_data', action='store_true')
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-data', action='store_true')
    parser.add_argument('--latex', action='store_true')
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('--select-mlm', action='store_true')
    parser.add_argument('--data-avg', action='store_true')
    # parser.add_argument('--best-mlm', action='store_true')
    # parser.add_argument('--language', default='groningen')
    args = parser.parse_args()

    # if args.best_pos and (args.best_mlm or args.pos_ckpt is not None):
    #     print('double pos ckpt identifier')
    #     exit(1)
    # if args.best_mlm and (args.best_pos or args.mlm_ckpt is not None):
    #     print('double mlm ckpt identifier')
    #     exit(1)

    if args.all_data:
        args.ungroup = True

    if args.plot:
        args.lang = 'groningen'
        args.test = True
    if args.plot_data:
        args.test = True
    if args.test:
        args.best = True

    # if args.latex:
    #     matplotlib.use("pgf")
    #     matplotlib.rcParams.update({
    #         "pgf.texsystem": "pdflatex",
    #         'font.family': 'serif',
    #         'text.usetex': True,
    #         'pgf.rcfonts': False,
    #     })

    data, data_names = load_results(args.name, args.lang, args)
    table, headers = create_table(data, data_names, args)

    if args.select_mlm:
        select_best_mlm(table, headers)

    if args.plot:
        fy_data, fy_data_names = load_results(args.name, 'frisian', args)
        fy_table, fy_headers = create_table(fy_data, fy_data_names, args)
        assert fy_data_names == data_names
        assert fy_headers == headers
        assert len(fy_table) == len(table)
        for i in range(len(table)):
            table[i][8] = fy_table[i][8]

        base_name = f'{args.name.split("_")[0]}_None'
        base_data, base_data_names = load_results(base_name, args.lang, args)
        base_table, base_headers = create_table(base_data, base_data_names,
                                                args)
        assert base_data_names == data_names
        assert base_headers == headers
        assert len(base_table) == len(table)

        make_plots(headers, table, base_table, args)

    if args.plot_data:
        make_data_plots(headers, table, args)

    if args.data_avg:
        agg_table = {}
        for row in table:
            model, lang, pos, pos_ckpt, mlm, mlm_ckpt = row[:6]
            scores = row[6:]
            name = (model, lang, mlm)
            if name in agg_table:
                scores = [(s1 + s2) / 2
                          for s1, s2 in zip(agg_table[name], scores)]
            agg_table[name] = scores
        agg_table = [list(name) + scores for name, scores in agg_table.items()]

        for i in range(3, len(headers) - 3):
            print(i)
            ckpt_headers = []
            pivot_table = []
            prev_name = (None, None)

            for row in agg_table:
                model, lang, mlm = row[:3]
                score = row[i]
                if score is None or score == 0:
                    continue

                if prev_name != (model, lang):
                    pivot_table.append([lang, model] +
                                       [None] * len(ckpt_headers))
                    prev_name = model, lang
                if mlm not in ckpt_headers:
                    ckpt_headers.append(mlm)
                    for row2 in pivot_table:
                        row2.append(None)
                pivot_table[-1][2 + ckpt_headers.index(mlm)] = score

            pivot_table = sorted(pivot_table)
            print(
                '\n',
                tabulate(pivot_table,
                         headers=['lang', 'model'] + ckpt_headers,
                         floatfmt='.3f',
                         tablefmt='latex' if args.latex else 'simple'))


if __name__ == '__main__':
    main()
