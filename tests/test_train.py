"""train.pyのテストコード"""

import pytest
import torch
from lecture_final.train import Net


def test_net_forward():
    """Netクラスのforwardメソッドをテスト"""
    model = Net()
    # ダミー入力 (batch_size=2, channels=1, height=28, width=28)
    x = torch.randn(2, 1, 28, 28)
    output = model(x)

    # 出力形状の確認
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"

    # 対数確率の総和が1に近いか確認（log_softmaxの性質）
    probs = torch.exp(output)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_net_output_range():
    """Netの出力が対数確率の範囲内にあるかテスト"""
    model = Net()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)

    # log_softmaxの出力は負の値
    assert (output <= 0).all(), "Log probabilities should be <= 0"


def test_net_initialization():
    """Netクラスが正しく初期化されるかテスト"""
    model = Net()

    # レイヤーの存在確認
    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2")
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")

    # パラメータ数の確認（大まかに）
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
