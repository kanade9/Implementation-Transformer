import torch.nn as nn
from torchtext.vocab import Vectors


# Embedderモジュールの実装

# idで管理されている単語をベクトル表現で表す
class Embedder(nn.module):

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)
        # freeze=Trueでバックプロパゲーションの更新を防ぐ

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec

# 動作確認
from utils.dataloader imoort　get_IMDb_DataLoaders_and_TEXT

train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(
    max_length=256, batch_size=24)
)

batch = next(iter(train_dl))

net1 = Embedder(TEXT.vocab.vectors)

x = batch.Text[0]
x1 = net1(x)

print("入力テンソルサイズ:", x.shape())
print("出力テンソルサイズ:", x.shape())
