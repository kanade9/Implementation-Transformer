import torch.nn as nn, torch, math
from torchtext.vocab import Vectors
import ConvertTsv


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


# PositionalEncodeモジュールの実装
# 入力単語の位置情報のベクトル情報付加
class PositionalEncoder(nn.module):

    # d_model は単語ベクトルの次元数を表す
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model

        # 単語の順番posと埋め込みベクトルの次元の位置(i)によって定まる値の表をpeとして作成

        pe = torch.zeros(max_seq_len, d_model)

        # GPU使う用コード
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        # ここも原著論文読んで後には理解したい　(Attention is all you need)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))

        # peの先頭にミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配は計算しない
        self.pe.requires_grad = False

    def forward(self, x):
        # 入力xとPositional Encodingの足し算
        # 入力はpeよりも小さくなってしまうので、大きくする

        ret = math.sqrt(self.d_model) * x + self.pe

        return ret


# 動作確認(Embedder)

train_dl, val_dl, test_dl, TEXT = ConvertTsv.get_IMDb_DataLoaders_and_TEXT(
    max_length=256, batch_size=24, debug_log=True)

batch = next(iter(train_dl))

net1 = Embedder(TEXT.vocab.vectors)

x = batch.Text[0]
x1 = net1(x)

print("動作確認(Embedder)")
print("入力テンソルサイズ:", x.shape())
print("出力テンソルサイズ:", x.shape())

# 動作確認(Positional Encoder)

net1=Embedder(TEXT.vocab.vectors)
net2=PositionalEncoder(d_model=300,max_seq_len=256)

x = batch.Text[0]
x1=net1(x)
x2=net2(x1)

print("動作確認(Positional Encoder)")
print("入力テンソルサイズ:", x.shape())
print("出力テンソルサイズ:", x.shape())
