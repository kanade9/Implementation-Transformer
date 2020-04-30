# トレーニングデータのneg,pos,テストデータのneg,posのtsvファイル作成
import glob, os, io, string, urllib.request, tarfile, re, torchtext, random, zipfile
from torchtext.vocab import Vectors

# 日本語を分類するにあたって追加
import glob, torch, mojimoji, pandas as pd, numpy,itertools
from natto import MeCab
from sklearn.model_selection import train_test_split
from typing import List

List[str]

base = os.path.dirname(os.path.abspath(__file__))

mecab = MeCab("-Owakati")
mecab_sym = MeCab("-Ochasen")


# mecab
def tokenizer_with_preprocessing(text: str) -> List[str]:
    # torchtextDataFieldに引数taggerをうまく渡せなかったのでここに書いた。
    tagger=mecab
    return [tok for tok in tagger.parse(text).split()]

# mecab_sym
def pick_sym(text: str, tagger: MeCab) -> List[str]:
    node = tagger.parse(text)
    node_list = node.split("\n")
    return [node[0] for node in node_list if "記号" in node]


def del_sym(df_text: str, symbol_list: List) -> str:
    trans_table = str.maketrans({"０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",})

    df_text.translate(trans_table)
    df_text.replace('\d+年', '')
    df_text.replace('\d+月', '')
    df_text.replace('\d+日', '')
    df_text.replace('\d', '')
    df_text.replace('\n', '')

    for i in range(len(symbol_list)):
        if not symbol_list[i]:
            continue
        df_text.replace(symbol_list[i], '')
    return df_text


def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24, debug_log=False):
    filenames = glob.glob(base + '/data-japanese/text/*/*')

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

    print(mecab_sym.parse('こんにちはかなちゃん。'))
    a = 'こんにちはかなちゃん。'
    print(mecab_sym.parse(a).split("\n"))

    news['text'] = news['text'].apply(mojimoji.han_to_zen)

    symbol_list = list(news["text"].apply(pick_sym,tagger=mecab_sym))

    # 2次元のsymbol_listを1次元リストにする。
    all_symbol_list = itertools.chain.from_iterable(symbol_list)
    # for i in range(len(news)):
    # symbol_list.append(pick_sym(news['text'].iloc[i]))

    all_symbol_list = list(set(all_symbol_list))
    print(all_symbol_list)

    # いらないものを取り去り、tsvに格納する作業
    news['text'] = news['text'].apply(del_sym, symbol_list=all_symbol_list)
    news['label'] = news['label'].replace({'it-life-hack': 0, 'kaden-channel': 1, })

    train_set, test_set = train_test_split(news, test_size=0.2)

    """
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
    """
    train_set[['text', 'label']].to_csv(base + '/data-japanese/jp_train.tsv', sep='\t')
    test_set[['text', 'label']].to_csv(base + '/data-japanese/jp_test.tsv', sep='\t')

    # ここから前処理

    print(tokenizer_with_preprocessing(text='私はお寿司が好きです。'))

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
    jp_word2vec_vectors = Vectors(name=base + '/data-japanese/japanese_word2vec_vectors.vec')

    if debug_log:
        print("1単語を表現する次元数:", jp_word2vec_vectors.dim)
        print("単語数:", len(jp_word2vec_vectors.itos))

    # ベクトル化したバージョンのボキャブラリーを作成する
    TEXT.build_vocab(train_ds, vectors=jp_word2vec_vectors, min_freq=1)

    # ボキャブラリーのベクトルの確認を行う
    if debug_log:
        print(TEXT.vocab.vectors.shape)
        # print(TEXT.vocab.vectors)
        # print(TEXT.vocab.stoi)

    # DataLoaderの作成
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

    # 検証データセットで確認
    batch = next(iter(train_dl))

    if debug_log:
        print(batch.Text)
        print(batch.Label)
    return train_dl, val_dl, test_dl, TEXT


get_IMDb_DataLoaders_and_TEXT(debug_log=1)
