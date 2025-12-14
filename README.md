# lecture_final
improved-pytorch-mnist

# MNIST Classification with Improved Code Quality

本プロジェクトは、PyTorch公式のMNISTサンプルコードに対して、現代的なPython開発手法を適用し、コード品質を向上させたものです。

## 改善内容

- uvによるパッケージ管理
- 型ヒントの追加
- Docstringの整備
- pytestによる自動テスト
- Gitブランチ戦略の実践

## セットアップ
```bash
# リポジトリのクローン
git clone https://github.com/zh541556051/lecture_final.git
cd lecture_final

# 依存関係のインストール
uv sync

# 仮想環境の有効化
source .venv/bin/activate
```

## テストの実行
```bash
pytest tests/ -v
```

## 元のコード

元のコード：[PyTorch MNIST Example](https://github.com/pytorch/examples/blob/main/mnist/main.py)