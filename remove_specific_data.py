import re,os,glob

base = os.path.dirname(os.path.abspath(__file__))
filenames = glob.glob(base + '/data-japanese/text/*/*')

for filename in filenames:
    # print(filename)
    if re.search('LICENSE', filename):
        continue
    with open(filename, 'r', encoding='utf-8') as f:
        filepath, basename = os.path.split(filename)
        text=f.readlines()
        if "キンドルが日本に参入する" in str(text) and "据え置きゲーム機" in str(text):
            print("Found")
            print(filename)
            exit
