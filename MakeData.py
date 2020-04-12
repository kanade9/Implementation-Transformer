# for colab
"""
! mkdir data
"""

# Download IMDb dataset
import glob, os, io, string, urllib.request, tarfile, re, torchtext, random, zipfile
from torchtext.vocab import Vectors

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
save_path = "./data/aclImdb_v1.tar.gz"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
urllib.request.urlretrieve(url)
tar = tarfile.open('./data/aclImdb_v1.tar.gz')
tar.extractall('./data/')
tar.close()

# 単語ベクトルのダウンロード 5ふんぐらいかかる
url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
save_path = "./data/wiki-news-300d-1M.vec.zip"
if not os.path.exists(save_path):
    urllib.request.urlretrieve(url, save_path)
zip = zipfile.ZipFile("./data/wiki-news-300d-1M.vec.zip")
zip.extractall("./data/")  # ZIPを解凍
zip.close()
