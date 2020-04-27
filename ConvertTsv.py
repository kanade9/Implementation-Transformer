# トレーニングデータのneg,pos,テストデータのneg,posのtsvファイル作成
import glob, os, io, string, urllib.request, tarfile, re, torchtext, random, zipfile
from torchtext.vocab import Vectors
import glob, mojimoji


def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24, debug_log=False):
    filenames = glob.glob('text/*/*')

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

    mecab_sym = MeCab.Tagger('-Ochasen')
    symbol_list = []

    def pick_sym(text):
        node = mecab_sym.parseToNode(text)
        while node:
            if node.feature.split(',')[0] == '記号':
                symbol_list.append(node.feature.split(",")[6])
            node = node.next

    news['text'] = news['text'].apply(mojimoji.han_to_zen)

    symbol_list = []
    for i in range(len(news)):
        symbol_list.append(pick_sym(news['text'].iloc[i]))

    symbol_list = list(set(symbol_list))

    def del_sym(df):
        df['text'] = df['text'].str.replace('\d+年', '', regex=True)
        df['text'] = df['text'].str.replace('\d+月', '', regex=True)
        df['text'] = df['text'].str.replace('\d+日', '', regex=True)
        df['text'] = df['text'].str.replace('\d+', '0', regex=True)
        df['text'] = df['text'].str.replace('\n', '')
        for i in range(len(symbol_list)):
            df['text'] = df['text'].str.replace(symbol_list[i], '')
        return df

    # いらないものを取り去り、tsvに格納する作業
    for DataFrameTextIndex in range(len(news)):
        news['text'].iloc[DataFrameTextIndex] = del_sym(news['text'].iloc[DataFrameTextIndex])

    news_delsym['label'] = news_delsym['label'].replace({'it-life-hack': 0, 'kaden-channel': 1, })
    train_set, test_set = train_test_split(news_delsym, test_size=0.2)

    f = open('./data-japanese/jp_train.tsv', 'w')
    for index in range(len(train_set)):
        text = train_set(news['text'].iloc[index])
        label_num = train_set(news['label'].iloc[index])
        text = text + '\t' + str(label_num) + '\t' + '\n'
        f.write(text)
    f.close()

    f = open('./data-japanese/jp_test.tsv', 'w')
    for index in range(len(test_set)):
        text = test_set(news['text'].iloc[index])
        label_num = test_set(news['label'].iloc[index])
        text = text + '\t' + str(label_num) + '\t' + '\n'
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
        path='./data-japanese/', train='jp_train.tsv',
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
    jp_word2vec_vectors = Vectors(name='data-japanese/japanese_word2vec_vectors.vec')

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
