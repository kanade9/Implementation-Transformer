# for colab
"""
! mkdir data
"""

# Download IMDb dataset
import glob, os, io, string, urllib.request, tarfile

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
urllib.request.urlretrieve(url)
tar = tarfile.open('./data/aclImdb_v1.tar.gz')
tar.extractall('./data/')
tar.close()

# トレーニングデータのneg,pos,テストデータのneg,posのtsvファイル作成


path_list = ['.data/aclImdb/train/pos/', '.data/aclImdb/train/neg', '.data/aclImdb/test/pos/',
             '.data/aclImdb/test/neg/']

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
