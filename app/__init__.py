"""北九州市ごみ分別RAGチャットボット。

`app.data`(データ層)と `app.rag`(応用層)を含む。`server.py` は本パッケージの
一部だが、systemd からは作業ディレクトリ `app/` で `uvicorn server:app` として
起動されるため、`server.py` 側でリポジトリルートを sys.path に追加している。
"""
