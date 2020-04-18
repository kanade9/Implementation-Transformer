import torch.nn as nn
import torch.optim as optim
from ConvertTsv import get_IMDb_DataLoaders_and_TEXT

# 読み込みを行う
train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=64)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dl, "val": val_dl}

from Transformer import TransformerClassification

# モデル構築
net = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)


# ネットワークの初期化を定義
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Linear層の初期化
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# 訓練モードに設定
net.train()

# TransformerBlockモジュールを初期化実行
net.net3_1.apply(weights_init)
net.net3_2.apply(weights_init)

print('ネットワーク準備OK')

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmaxを計算してからnn.NLLLoss(negative log likelihood loss)を計算

# 最適化手法の設定
learning_rate = 2e-5
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
