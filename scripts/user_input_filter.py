import tkinter as tk
import pandas as pd
import numpy as np
from PIL import Image, ImageTk

emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def user_verification(df):
    print("number of images to verify:", len(df))
    root = tk.Tk()
    root.title("Face Verification")
    # List to hold labels
    labels = []
    
    for index, row in df.iterrows():
        print("Index:", index)
        print("Emotion:", row['emotion'])
        image = row['pixels']
        image = np.array(image.split(), dtype='uint8').reshape(48, 48)
        img = Image.fromarray(image)
        # Resize the image to fit the label
        img = img.resize((240, 240))  # Adjust size as needed
        img = ImageTk.PhotoImage(img)
        
        label = tk.Label(root, image=img)
        label.image = img 
        label.pack(side=tk.LEFT, padx=5, pady=5)  # Pack labels horizontally
        
        # Append the label to the list for later deletion
        labels.append(label)

        print("Emotion:", emotion_classes[row['emotion']])
        print("number of images left to verify:", len(df))

        # take user input to decide if the image should be kept or not
        keep = input("Is this a face? (y/n): ")
        if keep.lower() == 'y':
            df = df.drop(index=index)
        
        # q ro quit the loop
        if keep.lower() == 'q':
            break
        # hide image after user input
        label.destroy()
        labels[-1].destroy()
    
    return df

df = pd.read_csv('../data/pics_without_faces_user.csv')
df = user_verification(df)
df.to_csv('../data/pics_without_faces_user.csv', index=False)
print(len(df), "pictures saved to pics_without_faces_user.csv")

df = pd.read_csv('../data/train.csv')
print("df:", df.info(), df.head())

df_pics_to_rm = pd.read_csv('../data/pics_without_faces_user.csv')
print("df_pics_to_rm:", df_pics_to_rm.info(), df_pics_to_rm.head())

# drop the rows from train.csv that were saved to pics_without_faces_user.csv
print("df before drop:", df.info())
df = df.drop(df_pics_to_rm['original_index'])
print("df after drop:", df.info())

# write df to the end product: train_filtered.csv
df.to_csv('../data/train_filtered.csv', index=False)