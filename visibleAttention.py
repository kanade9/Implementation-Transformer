# HTMLの作成

def highlight(word,attn):
    html_color='#%02X%02X%02X' % ( 255,int(255*(1-attn)),int(255*(1-attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color,word)


def mk_html(index,batch,preds,normlized_weight_1,normlized_weight_2, TEXT):
    # indexの結果を抽出
    sentence=batch.Text[0][index] # 文書
    label=batch.Label[index] # ラベル
    pred = preds[index] # 予測

    # indexのAttentionを抽出と規格化
    # これは0番目の<cls>のAttention
    attens1 = normlized_weight_1[index,0,:]
    attens1 /= attens1.max()

    attens2 = normlized_weight_2[index,0,:]
    attens2 /= attens2.max()

    # ラベルと予測結果を文字に置き換える
    if label ==0:
        label_str="it-life-hack"
    else:
        label_str="kaden-channel"

    if pred ==0:
        pred_str="it-life-hack"
    else:
        pred_str="kaden-channel"

    html='正解ラベル:{}<br>推論ラベル:{}<br><br>'.format(label_str,pred_str)

    # 1段目のattention
    html += '[TransformerBlockの1段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence,attens1):
        html+= highlight(TEXT.vocab.itos[word],attn)

    html+="<br><br>"

    # 2段目のattention

    html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence,attens2):
        html+= highlight(TEXT.vocab.itos[word],attn)

    html+="<br><br>"

    return html

