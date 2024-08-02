"""便利なJSON出力ファイルを生成する"""

from tqdm import tqdm  # プログレスバーを表示するためのライブラリ
from typing import List  # 型ヒントのためのライブラリ
import pandas as pd  # データ操作のためのライブラリ
from langchain.chat_models import ChatOpenAI  # 言語モデルを利用するためのライブラリ
import json  # JSON操作のためのライブラリ


def aggregation(config):
    # 出力ファイルのパスを設定
    path = f"outputs/{config['output_dir']}/result.json"

    # 結果を格納する辞書を初期化
    results = {
        "clusters": [],
        "comments": {},
        "translations": {},
        "overview": "",
        "config": config,
    }

    # 引数のCSVファイルを読み込み
    arguments = pd.read_csv(f"outputs/{config['output_dir']}/args.csv")
    arguments.set_index('arg-id', inplace=True)

    # コメントのCSVファイルを読み込み
    comments = pd.read_csv(f"inputs/{config['input']}.csv")
    useful_comment_ids = set(arguments['comment-id'].values)
    
    # argumentsで参照されているコメントIDのみを抽出し、results辞書に格納
    for _, row in comments.iterrows():
        id = row['comment-id']
        if id in useful_comment_ids:
            res = {
                'comment': row['comment-body'],
                #'categoryLabel': row['labels'],
                #'kutikomi_unique':row['kutikomi_unique']
                }
            numeric_cols = ['agrees', 'disagrees']
            string_cols = ['video', 'interview', 'timestamp']
            for col in numeric_cols:
                if col in row:
                    res[col] = float(row[col])
            for col in string_cols:
                if col in row:
                    res[col] = row[col]
            results['comments'][str(id)] = res

    # オブション：翻訳の設定を取得し、翻訳結果を読み込み
    languages = list(config.get('translation', {}).get('languages', []))
    if len(languages) > 0:
        with open(f"outputs/{config['output_dir']}/translations.json") as f:
            translations = f.read()
        results['translations'] = json.loads(translations)


    # labelsファイルの修正
    label_df = pd.read_csv(f"outputs/{config['output_dir']}/labels.csv")
    similarity_verbalization_df = pd.read_csv(f"outputs/{config['output_dir']}/similarity_verbalization.csv")
    difference_verbalization_df = pd.read_csv(f"outputs/{config['output_dir']}/difference_verbalization.csv")
    merged_df = label_df.merge(similarity_verbalization_df, on='cluster-id', how='inner')
    merged_df = merged_df.merge(difference_verbalization_df, on='cluster-id', how='inner')
    merged_df.to_csv(f"outputs/{config['output_dir']}/labels.csv", index=False)


    # クラスター、ラベル、テイクアウェイのCSVファイルを読み込み
    clusters = pd.read_csv(f"outputs/{config['output_dir']}/clusters.csv")
    labels = pd.read_csv(f"outputs/{config['output_dir']}/labels.csv")
    takeaways = pd.read_csv(f"outputs/{config['output_dir']}/takeaways.csv")
    takeaways.set_index('cluster-id', inplace=True)

    # 概要テキストを読み込み
    with open(f"outputs/{config['output_dir']}/overview.txt") as f:
        overview = f.read()
    results['overview'] = overview

    # ラベルごとにクラスターを処理し、結果に追加
    for _, row in labels.iterrows():
        cid = row['cluster-id']
        label = row['label']
        # label.csvの該当カラムに変更する
        cluster_chigai =  row['difference']
        cluster_ruizi = row['similarity']
        arg_rows = clusters[clusters['cluster-id'] == cid]

        arguments_in_cluster = []
        for _, arg_row in arg_rows.iterrows():
            arg_id = arg_row['arg-id']
            argument = arguments.loc[arg_id]['argument']
            comment_id = arguments.loc[arg_id]['comment-id']
            categoryLabel = arguments.loc[arg_id]['categoryLabel'] #追加
            kutikomi_unique = arguments.loc[arg_id]['kutikomi_unique']
            koushiki_unique = arguments.loc[arg_id]['koushiki_unique'] 
            x = float(arg_row['x'])
            y = float(arg_row['y'])
            #p = float(arg_row['probability'])
                        # 'probability'が存在しない場合の代替処理
            if 'probability' in arg_row:
                p = float(arg_row['probability'])
            else:
                p = 1.0  
            obj = {
                'arg_id': arg_id,
                'argument': argument,
                'comment_id': str(comment_id),
                'categoryLabel': int(categoryLabel),
                'kutikomi_unique': int(kutikomi_unique),
                'koushiki_unique': int(koushiki_unique),
                'x': x,
                'y': y,
                'p': p,
            }
            arguments_in_cluster.append(obj)
        results['clusters'].append({
            'cluster': label,
            'cluster_chigai': cluster_chigai,
            'cluster_ruizi': cluster_ruizi, 
            'cluster_id': str(cid),
            'takeaways': takeaways.loc[cid]['takeaways'],
            'arguments': arguments_in_cluster
        })

    # 結果をJSONファイルに書き込み
    with open(path, 'w') as file:
        json.dump(results, file, indent=2)
