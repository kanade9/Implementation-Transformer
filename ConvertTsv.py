# トレーニングデータのneg,pos,テストデータのneg,posのtsvファイル作成
import glob, os, io, string, urllib.request, tarfile, re, torchtext, random, zipfile
from torchtext.vocab import Vectors

# 日本語を分類するにあたって追加
import glob, torch, mojimoji, pandas as pd, numpy
from natto import MeCab
from sklearn.model_selection import train_test_split

base = os.path.dirname(os.path.abspath(__file__))


# print(base)


def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24, debug_log=False):
    filenames = glob.glob(base + '/data-japanese/text/*/*')
    # filenames = [base + '/data-japanese/text/kaden-channel/kaden-channel-5774093.txt',
    #              base + '/data-japanese/text/kaden-channel/kaden-channel-5774562.txt']
    # print(filenames,len(filenames))
    keys = []
    texts = []
    labels = []
    for filename in filenames:
        if re.search('LICENSE', filename):
            continue

        with open(filename, 'r', encoding='utf-8') as f:
            filepath, basename = os.path.split(filename)

            keys.append(os.path.splitext(basename)[0])
            texts.append(''.join(f.readlines()[2:]))
            labels.append(os.path.split(filepath)[-1])

    news = pd.DataFrame({'key': keys, 'label': labels, 'text': texts})

    # natoo-pyではコンストラクタに渡す

    mecab_sym = MeCab("-Ochasen")
    print(mecab_sym.parse('こんにちはかなちゃん。'))
    a = 'こんにちはかなちゃん。'
    print(mecab_sym.parse(a).split("\n"))
    symbol_list = []

    def pick_sym(text):
        node = mecab_sym.parse(text)
        node_list = node.split("\n")
        for node in node_list:
            if '記号' in node:
                symbol_list.append(node[0])

    news['text'] = news['text'].apply(mojimoji.han_to_zen)

    symbol_list = []
    for i in range(len(news)):
        symbol_list.append(pick_sym(news['text'].iloc[i]))

    symbol_list = list(set(symbol_list))
    print(symbol_list)

    def del_sym(df):
        print(df['text'])
        df['text'] = df['text'].replace('\d+年', '')
        df['text'] = df['text'].replace('\d+月', '')
        df['text'] = df['text'].replace('\d+日', '')
        df['text'] = df['text'].replace('\d', '')
        df['text'] = df['text'].replace('\n', '')

        for i in range(len(symbol_list)):
            if not symbol_list[i]:
                continue
            df['text'] = df['text'].replace(symbol_list[i], '')
        return df

    # いらないものを取り去り、tsvに格納する作業
    for DataFrameTextIndex in range(len(news)):
        print(news.iloc[DataFrameTextIndex])
        news.iloc[DataFrameTextIndex] = del_sym(news.iloc[DataFrameTextIndex])

    news['label'] = news['label'].replace({'it-life-hack': 0, 'kaden-channel': 1, })

    # train_set, test_set = train_test_split(news, test_size=0.2)
    train_set, test_set = train_test_split(news, test_size=0.2)

    f = open(base + '/data-japanese/jp_train.tsv', 'w')
    for index in range(len(train_set)):
        text = train_set['text'].iloc[index]
        label_num = train_set['label'].iloc[index]
        text = text + '\t' + str(label_num) + '\t' + '\n'
        f.write(text)
    f.close()

    f = open(base + '/data-japanese/jp_test.tsv', 'w')
    for index in range(len(test_set)):
        text = test_set['text'].iloc[index]
        label_num = test_set['label'].iloc[index]
        text = text + '\t' + str(label_num) + '\t' + '\n'
        f.write(text)
    f.close()

    # ここから前処理
    mecab = MeCab("-Owakati")

    def tokenizer_with_preprocessing(text):
        return [tok for tok in mecab.parse(text).split()]

    print(tokenizer_with_preprocessing('私はお寿司が好きです。'))

    # DataLoaderの作成
    # init_token 全部の文章で文頭に入れておく単語
    # eos_token 全部の文章で文末に入れておく単語

    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                                use_vocab=True, lower=True, include_lengths=True, batch_first=True,
                                fix_length=max_length, init_token="<cls>", eos_token="<eos>")

    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # Datasetの作成
    # 訓練および検証データセットを分ける。
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=base + '/data-japanese/', train='jp_train.tsv',
        test='jp_test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    if debug_log:
        print('訓練および検証のデータの数', len(train_val_ds))
        print('1つ目の訓練および検証のデータ', vars(train_val_ds[0]))

    # 訓練データ:検証データを8:2で分ける。
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))
    # これで、訓練、検証、テスト3つのDatasetの作成が完了.

    # debug
    if debug_log:
        print('訓練データの数', len(train_ds))
        print('検証データの数', len(val_ds))
        print('１つ目の訓練データ', vars(train_ds[0]))

    # torchtextで単語ベクトルとして英語学習済みモデルを利用する。
    jp_word2vec_vectors = Vectors(name=base+'/data-japanese/japanese_word2vec_vectors.vec')

    if debug_log:
        print("1単語を表現する次元数:", jp_word2vec_vectors.dim)
        print("単語数:", len(jp_word2vec_vectors.itos))

    # ベクトル化したバージョンのボキャブラリーを作成する
    TEXT.build_vocab(train_ds, vectors=jp_word2vec_vectors, min_freq=10)

    # ボキャブラリーのベクトルの確認を行う
    if debug_log:
        print(TEXT.vocab.vectors.shape)
        print(TEXT.vocab.vectors)
        print(TEXT.vocab.stoi)

    # DataLoaderの作成
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

    # 検証データセットで確認
    batch = next(iter(val_dl))

    if debug_log:
        print(batch.Text)
        print(batch.Label)

    return train_dl, val_dl, test_dl, TEXT


get_IMDb_DataLoaders_and_TEXT(debug_log=bool(1))
