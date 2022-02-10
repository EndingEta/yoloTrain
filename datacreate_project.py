import os
import ast
import glob
from pathlib import Path
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
import shutil



strIMAGES= 'image'
strLabels = 'labels'

image_cart = glob.glob('cart_items/*.jpg')
label_cart = glob.glob('cart_items/*.txt')
print(len(image_cart), len(label_cart))


dataset = {strIMAGES: [], strLabels: []}

img_all = image_cart
label_all = label_cart

print(len(img_all), len(label_all))

def processdata(data, datatype):

    for index, row in tqdm(data.iterrows(), total=len(data)):
        image_id = row[strIMAGES]
        txt_id = row[strLabels]
        new_image = Path(image_id).stem
        new_txt = Path(txt_id).stem
        shutil.copyfile(
            image_id,
            f'cart_dataset/images/{datatype}/{new_image}.jpg'
        )
        shutil.copyfile(
            txt_id,
            f'cart_dataset/labels/{datatype}/{new_txt}.txt'
        )

if __name__ == '__main__':
    for ID, i in enumerate(img_all):
        dataset[strIMAGES].append(img_all[ID])
        dataset[strLabels].append(label_all[ID])

    new = pd.DataFrame(data=dataset)
    print(new)

    df_train, df_test = model_selection.train_test_split(
        new,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    df_train = df_train.reset_index(drop=True)
    df_test =df_test.reset_index(drop=True)

    processdata(df_train, "train")
    processdata(df_test, "val")
