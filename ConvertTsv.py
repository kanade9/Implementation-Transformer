# トレーニングデータのneg,pos,テストデータのneg,posのtsvファイル作成
import glob, os, io, string, urllib.request, tarfile, re, torchtext, random, zipfile
from torchtext.vocab import Vectors


def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24, debug_log=False):
    path_list = ['./data/aclImdb/train/pos/', './data/aclImdb/train/neg/', './data/aclImdb/test/pos/',
                 './data/aclImdb/test/neg/']

    base = os.path.dirname(os.path.abspath(__file__))
    f = open(base+'/data/IMDb_train.tsv', 'w')

    path = base+'/data/aclImdb/train/pos/'

    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            text = text.replace('\t', " ")
            text = text + '\t' + '1' + '\t' + '\n'
            f.write(text)

    path = base+'/data/aclImdb/train/neg/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            text = text.replace('\t', " ")
            text = text + '\t' + '0' + '\t' + '\n'
            f.write(text)

    f.close()

    f = open(base+'/data/IMDb_test.tsv', 'w')

    path = base+'/data/aclImdb/test/pos/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            text = text.replace('\t', " ")
            text = text + '\t' + '1' + '\t' + '\n'
            f.write(text)

    path = base+'/data/aclImdb/test/neg/'

    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            text = text.replace('\t', " ")
            text = text + '\t' + '0' + '\t' + '\n'
            f.write(text)

    f.close()

    # ここから前処理
    def preprocessing_text(text):

        text = re.sub('<br />', '', text)

        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
        text = text.replace(p, " ")

        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        return text

    # 分かち書き
    def tokenizer_punctuation(text):

        return text.strip().split()

    def tokenizer_with_preprocessing(text):

        text = preprocessing_text(text)
        ret = tokenizer_punctuation(text)
        return ret

    print(tokenizer_with_preprocessing('I like cats.'))

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
        path=base+'/data/', train='IMDb_train.tsv',
        test='IMDb_test.tsv', format='tsv',
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
    english_fasttext_vectors = Vectors(name=base+'/data/wiki-news-300d-1M.vec')

    if debug_log:
        print("1単語を表現する次元数:", english_fasttext_vectors.dim)
        print("単語数:", len(english_fasttext_vectors.itos))

    # ベクトル化したバージョンのボキャブラリーを作成する
    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

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

get_IMDb_DataLoaders_and_TEXT(debug_log=1)