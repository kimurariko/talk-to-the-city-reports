"""Cluster the arguments using UMAP + HDBSCAN and GPT-4."""

# 必要なライブラリをインポート
import pandas as pd
import numpy as np
from importlib import import_module

def clustering(config):
    # データセットのパスを設定
    dataset = config['output_dir']
    path = f"outputs/{dataset}/clusters.csv"

    # 引数のデータを読み込む
    arguments_df = pd.read_csv(f"outputs/{dataset}/args.csv")
    arguments_array = arguments_df["argument"].values

    # 埋め込みデータを読み込む
    embeddings_df = pd.read_pickle(f"outputs/{dataset}/embeddings.pkl")
    embeddings_array = np.asarray(embeddings_df["embedding"].values.tolist())

    # クラスター数を設定
    clusters = config['clustering']['clusters']

    # クラスタリングを実行し結果を保存
    result = cluster_embeddings(
        docs=arguments_array,
        embeddings=embeddings_array,
        metadatas={
            "arg-id": arguments_df["arg-id"].values,
            "comment-id": arguments_df["comment-id"].values,
        },
        n_topics=clusters,
    )
    result.to_csv(path, index=False)

def cluster_embeddings(
    docs,
    embeddings,
    metadatas,
    min_cluster_size=2,
    n_components=2,
    n_topics=6,
):
    # 動的にモジュールをインポート（これにより、必要な場合にのみインポートされる）
    SpectralClustering = import_module('sklearn.cluster').SpectralClustering
    stopwords = import_module('nltk.corpus').stopwords
    HDBSCAN = import_module('hdbscan').HDBSCAN
    UMAP = import_module('umap').UMAP
    CountVectorizer = import_module('sklearn.feature_extraction.text').CountVectorizer
    BERTopic = import_module('bertopic').BERTopic

    # UMAPモデルを設定
    umap_model = UMAP(
        random_state=42,
        n_components=n_components,
    )

    # HDBSCANモデルを設定
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size)

    # ストップワードを設定
    stop = stopwords.words("english")

    # ベクトライザーモデルを設定
    vectorizer_model = CountVectorizer(stop_words=stop)

    # トピックモデルを設定
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )

    # トピックモデルをフィッティング：戻り値：各ドキュメントのトピック予測確率
    _, __ = topic_model.fit_transform(docs, embeddings=embeddings)

    # サンプル数と近傍数を設定
    n_samples = len(embeddings)
    n_neighbors = min(n_samples - 1, 10)

    # スペクトラルクラスタリングモデルを設定
    spectral_model = SpectralClustering(
        n_clusters=n_topics,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,  # 修正された近傍数を使用
        random_state=42
    )

    # UMAPで次元削減
    # ref:https://mathwords.net/fittransform
    umap_embeds = umap_model.fit_transform(embeddings)

    # スペクトラルクラスタリングを適用
    cluster_labels = spectral_model.fit_predict(umap_embeds)

    # クラスタリング結果を取得
    result = topic_model.get_document_info(
        docs=docs,
        metadata={
            **metadatas,
            "x": umap_embeds[:, 0],
            "y": umap_embeds[:, 1],
        },
    )

    # 結果のカラム名を小文字に変換
    result.columns = [c.lower() for c in result.columns]

    # 必要なカラムを選択し、クラスタIDを追加
    result = result[['arg-id', 'x', 'y', 'probability']]
    result['cluster-id'] = cluster_labels

    return result