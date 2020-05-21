# for colab
"""
! mkdir data
"""

# Download IMDb dataset
import glob, os, io, string, urllib.request, tarfile, re, torchtext, random, zipfile
from torchtext.vocab import Vectors
from gensim.models import KeyedVectors
url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
save_path = "./data-japanese/ldcc-20140209.tar.gz"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
urllib.request.urlretrieve(url)
tar = tarfile.open('./data-japanese/ldcc-20140209.tar.gz')
tar.extractall('./data-japanese/')
tar.close()
"""
# 単語ベクトルのダウンロード 10分ぐらいかかる
url = "http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2"
save_path = "./data-japanese/20170201.tar.bz2"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
tar2 = tarfile.open("./data-japanese/20170201.tar.bz2")
tar2.extractall("./data-japanese/")  # ZIPを解凍
tar2.close()

tmp_model = KeyedVectors.load_word2vec_format('./data-japanese/entity_vector/entity_vector.model.bin', binary=True)
tmp_model.wv.save_word2vec_format('./data-japanese/japanese_word2vec_vectors.vec')
"""