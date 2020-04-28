import torch.nn as nn, torch, math
import torch.nn.functional as F
from torchtext.vocab import Vectors
import ConvertTsv


# Embedderモジュールの実装

# idで管理されている単語をベクトル表現で表す
class Embedder(nn.Module):

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
class PositionalEncoder(nn.Module):

    # d_model は単語ベクトルの次元数を表す
    def __init__(self, d_model=200, max_seq_len=256):
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


# 論文では本来、マルチヘッドのAttentionを用いているが、今回は簡単のためにシングルヘッドAttentionを実装する
class Attention(nn.Module):
    def __init__(self, d_model=200):
        super().__init__()

        # 全結合層で特徴量を変換する
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換する
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.k_linear(v)

        # Attentionの値を計算する
        # 各値を足し算すると大きくなりすぎるため、root(d_k)で割って調整を行う
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # maskの計算
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化をする
        normlized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueと掛け算する
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換する
        output = self.out(output)

        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        # Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニット

        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization層(詳しくはpytorchのドキュメントを読む)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn = Attention(d_model)

        # Attentionの後の全結合二つ
        self.ff = FeedForward(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(x_normlized, x_normlized, x_normlized, mask)

        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights


"""
TransformerBlockモジュールを繰り返しやって特徴量変換をした後、Classificationモジュールに入力してネガポジ判定する。
init_token="<cls>"と設定して、先頭単語clsの特徴量を利用する。
この損失がback propagationで伝播してゆき。ネットワークが学習される。
"""


class ClassificationHead(nn.Module):
    def __init__(self, d_model=200, output_dim=2):
        super().__init__()

        # 全結合層 出力次元はneg,posの2次元
        self.linear = nn.Linear(d_model, output_dim)

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        # 各ミニバッチの各文の先頭の単語の特徴量(300次元)を取り出す
        x0 = x[:, 0, :]
        out = self.linear(x0)

        return out


# ここまでで実装したモジュールを組み合わせて分類タスク用のTransformerを実装する=最終的Transformerモデルのクラス
# Transformerでクラス分類
class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=200, max_seq_len=256, output_dim=2):
        super().__init__()

        # モデルの構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        # 単語をベクトルにする
        x1 = self.net1(x)
        # Position情報を足し算する
        x2 = self.net2(x1)
        # Self-Attentionで特徴量を変換する
        x3_1, normlized_weights_1 = self.net3_1(x2, mask)
        x3_2, normlized_weights_2 = self.net3_2(x3_1, mask)
        # 最終出力の0単語目を利用して、分類0-1のスカラーを出力する
        x4 = self.net4(x3_2)

        return x4, normlized_weights_1, normlized_weights_2


"""
# 動作確認(Embedder)

train_dl, val_dl, test_dl, TEXT = ConvertTsv.get_IMDb_DataLoaders_and_TEXT(
    max_length=256, batch_size=24, debug_log=True)

batch = next(iter(train_dl))

net1 = Embedder(TEXT.vocab.vectors)

x = batch.Text[0]
x1 = net1(x)

print("動作確認(Embedder)")
print("入力テンソルサイズ:", x.shape)
print("出力テンソルサイズ:", x1.shape)

# 動作確認(Positional Encoder)

net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)

x = batch.Text[0]
x1 = net1(x)
x2 = net2(x1)

print("動作確認(Positional Encoder)")
print("入力テンソルサイズ:", x1.shape)
print("出力テンソルサイズ:", x2.shape)

# 動作確認(TransformerBlock)
net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)
net3 = TransformerBlock(d_model=300)

# mask作成
x = batch.Text[0]
input_pad = 1  # 単語のIDでは'<pad>'が1
input_mask = (x != input_pad)
print(input_mask[0])

x1 = net1(x)
x2 = net2(x1)
# Self-Attentionで特徴量を変換する
x3, normlized_weights = net3(x2, input_mask)
print("TransformerBlock")
print("入力テンソルサイズ:", x2.shape)
print("出力テンソルサイズ:", x3.shape)
print("Attentionのサイズ", normlized_weights.shape)

# 最後にTransformer全体の動作確認を行う
batch = next(iter(train_dl))

net = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)

x = batch.Text[0]
input_mask = (x != input_pad)
out, normlized_weights_1, normlized_weights_2 = net(x, input_mask)

print("Transformer全体")
print("出力テンソルサイズ:", out.shape)
print("出力テンソルのsigmoid:", F.softmax(out, dim=1))

"""
