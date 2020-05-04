import torch
import torch.nn as nn
import torch.optim as optim
from ConvertTsv import get_IMDb_DataLoaders_and_TEXT
import time
import visibleAttention

start = time.time()
# 読み込みを行う
train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=64)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": train_dl, "val": val_dl}

from Transformer import TransformerClassification

# モデル構築
net = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, d_model=200, max_seq_len=256, output_dim=2)


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


# train
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス:", device)
    print('-----start-----')
    net.to(device)

    # ネットワークがある程度固定であれば高速化させる
    # 有効??
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッジを取り出すループ,batchはTextとLabelの辞書オブジェクト
            for batch in (dataloaders_dict[phase]):
                # GPU使用可ならGPUに送る
                inputs = batch.Text[0].to(device)  # 文書
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝播の計算
                with torch.set_grad_enabled(phase == 'train'):
                    # mask作成
                    input_pad = 1  # 最初の単語は'<pad>':1より
                    input_mask = (inputs != input_pad)

                    # Transformerに入力
                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, labels)  # 損失の計算
                    _, preds = torch.max(outputs, 1)  # ラベルの予測

                    # 訓練時は誤差逆伝播
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新する
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(
                'Epoch {}/{} | {:^5} | Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))
    return net


num_epoch = 10
net_trained = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epoch)
elapsed_time = time.time() - start
print("処理時間:{:.2f}".format(elapsed_time / 60) + "[分]")

# テストデータでの推論
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_trained.eval()
net_trained.to(device)

epoch_corrects = 0  # epochの正解数

# test_dlはtestデータのdataLoader
for batch in (test_dl):
    # batchはTextとLabelの辞書オブジェクト
    inputs = batch.Text[0].to(device)  # 文書
    labels = batch.Label.to(device)  # ラベル

    # 順伝播の計算
    with torch.set_grad_enabled(False):
        # mask作成
        input_pad = 1  # '<pad>':1より
        input_mask = (inputs != input_pad)

        # Transformerに入力する
        outputs, _, _ = net_trained(inputs, input_mask)
        _, preds = torch.max(outputs, 1)  # ラベルを予測する

        # 正解数の更新
        epoch_corrects += torch.sum(preds == labels.data)

epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

print('テストデータ{}個での正解率:{:.4f}'.format(len(test_dl.dataset), epoch_acc))

# ここから判定根拠のHTML出力を行う
output_data_num = 100
html_output = "<!DOCTYPE html><html lang=\"en\"><meta charset=\"utf-8\"/>"
f = open('./result.html', 'a')

for index in range(1, output_data_num + 1):
    batch = next(iter(test_dl))
    inputs = batch.Text[0].to(device)
    labels = batch.Label.to(device)

    input_pad = 1
    input_mask = (inputs != input_pad)

    # Transformerに入力する

    outputs, normlized_weights_1, normlized_weights_2 = net_trained(inputs, input_mask)

    _, preds = torch.max(outputs, 1)  # ラベルを予測

    html_output = visibleAttention.mk_html(index, batch, preds, normlized_weights_1, normlized_weights_2, TEXT)
    f.write(html_output)
f.close()
