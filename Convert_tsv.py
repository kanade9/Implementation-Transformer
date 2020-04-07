# for colab
"""
! mkdir data
"""

# Download IMDb dataset
import glob, os, io, string, urllib.request, tarfile, re, torchtext

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
urllib.request.urlretrieve(url)
tar = tarfile.open('./data/aclImdb_v1.tar.gz')
tar.extractall('./data/')
tar.close()

# トレーニングデータのneg,pos,テストデータのneg,posのtsvファイル作成


path_list = ['./data/aclImdb/train/pos/', './data/aclImdb/train/neg/', './data/aclImdb/test/pos/',
             './data/aclImdb/test/neg/']

for dataset_path in path_list:
    if 'train' in dataset_path:
        f = open('./data/IMDb_train.tsv', 'w')
    else:
        f = open('./data/IMDb_test.tsv', 'w')

    # ダウンロードしてきたファイルの整形を行う
    for fname in glob.glob(os.path.join(dataset_path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()
            text = text.replace('\t', " ")
            text = text + '\t' + '1' + '\t' + '\n'
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

max_length = 256
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                            use_vocab=True, lower=True, include_lengths=True, batch_first=True,
                            fix_length=max_length, init_token="<cls>", eos_token="<eos>")

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

# Datasetの作成
# 訓練および検証データセットを分ける。
train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='./data/', train='IMDb_train.tsv',
    test='IMDb_test.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)])
print('訓練および検証のデータの数',len(train_val_ds))
print('1つ目の訓練および検証のデータ', vars(train_val_ds[0]))
