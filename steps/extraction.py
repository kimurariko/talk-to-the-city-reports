import os
import json
from tqdm import tqdm
import pandas as pd
from langchain.chat_models import ChatOpenAI
from utils import messages, update_progress
import concurrent.futures

def extraction(config):
    # 出力ディレクトリとファイルパスの設定
    dataset = config['output_dir']
    path = f"outputs/{dataset}/args.csv"

    # 入力ファイルからコメントデータを読み込む
    comments = pd.read_csv(f"inputs/{config['input']}.csv")
    
    # モデル、プロンプト、ワーカー数、制限数の設定を取得
    model = config['extraction']['model']
    prompt = config['extraction']['prompt']
    workers = config['extraction']['workers']
    limit = config['extraction']['limit']

    # コメントIDを取得し、コメントIDをインデックスに設定
    comment_ids = (comments['comment-id'].values)[:limit]
    comments.set_index('comment-id', inplace=True)
    results = pd.DataFrame()
    update_progress(config, total=len(comment_ids))

    # コメントをバッチに分割して処理
    for i in tqdm(range(0, len(comment_ids), workers)):
        batch = comment_ids[i: i + workers]
        batch_inputs = [comments.loc[id]['comment-body'] for id in batch]
        batch_results = extract_batch(batch_inputs, prompt, model, workers)
        
        # バッチごとの結果をDataFrameに追加
        for comment_id, extracted_args in zip(batch, batch_results):
            for j, arg in enumerate(extracted_args):
                new_row = {"arg-id": f"A{comment_id}_{j}",
                           "comment-id": int(comment_id), "argument": arg}
                results = pd.concat(
                    [results, pd.DataFrame([new_row])], ignore_index=True)
        update_progress(config, incr=len(batch))
    
    # 結果をCSVファイルに保存
    results.to_csv(path, index=False)
    print(results.info())

def extract_batch(batch, prompt, model, workers):
    # スレッドプールを使用してバッチ処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(
            extract_arguments, input, prompt, model) for input in list(batch)]
        concurrent.futures.wait(futures)
        return [future.result() for future in futures]

def extract_arguments(input, prompt, model, retries=3):
    # LLM（大規模言語モデル）を使用して引数を抽出
    llm = ChatOpenAI(model_name=model, temperature=0.0)
    response = llm(messages=messages(prompt, input)).content.strip()
    try:
        # レスポンスをJSONとしてパース
        obj = json.loads(response)
        # 文字列の場合、リストに変換
        if isinstance(obj, str):
            obj = [obj]
        # 空文字列を除外
        items = [a.strip() for a in obj]
        items = filter(None, items)
        return items
    except json.decoder.JSONDecodeError as e:
        # JSONパースエラー時の処理
        print("JSON error:", e)
        print("Input was:", input)
        print("Response was:", response)
        if retries > 0:
            print("Retrying...")
            return extract_arguments(input, prompt, model, retries - 1)
        else:
            print("Silently giving up on trying to generate valid list.")
            return []
