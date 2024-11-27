import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

def atoi(s):
    return int(s)

outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data', outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

counters = {emotion: 0 for emotion in inner_names}
counters_test = {emotion: 0 for emotion in inner_names}

df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)
    emotion = inner_names[df['emotion'][i]]
    
    if i < 28709:
        img.save(f'data/train/{emotion}/im{counters[emotion]}.png')
        counters[emotion] += 1
    else:
        img.save(f'data/test/{emotion}/im{counters_test[emotion]}.png')
        counters_test[emotion] += 1

print("Training set size:", sum(counters.values()))
print("Validation set size:", sum(counters_test.values()))
print("Done!")
