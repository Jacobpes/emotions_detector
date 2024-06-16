import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# Creating the data for the table
data = {
    "Layer (type)": [
        "conv2d_1 (Conv2D)", "batchnorm_1 (BatchNormalization)", "conv2d_2 (Conv2D)", "batchnorm_2 (BatchNormalization)", 
        "maxpool2d_1 (MaxPooling2D)", "dropout_1 (Dropout)", "conv2d_3 (Conv2D)", "batchnorm_3 (BatchNormalization)", 
        "conv2d_4 (Conv2D)", "batchnorm_4 (BatchNormalization)", "maxpool2d_2 (MaxPooling2D)", "dropout_2 (Dropout)", 
        "conv2d_5 (Conv2D)", "batchnorm_5 (BatchNormalization)", "conv2d_6 (Conv2D)", "batchnorm_6 (BatchNormalization)", 
        "maxpool2d_3 (MaxPooling2D)", "dropout_3 (Dropout)", "flatten (Flatten)", "dense_1 (Dense)", 
        "batchnorm_7 (BatchNormalization)", "dropout_4 (Dropout)", "out_layer (Dense)"
    ],
    "Output Shape": [
        "(None, 48, 48, 512)", "(None, 48, 48, 512)", "(None, 48, 48, 256)", "(None, 48, 48, 256)", 
        "(None, 24, 24, 256)", "(None, 24, 24, 256)", "(None, 24, 24, 128)", "(None, 24, 24, 128)", 
        "(None, 24, 24, 128)", "(None, 24, 24, 128)", "(None, 12, 12, 128)", "(None, 12, 12, 128)", 
        "(None, 12, 12, 256)", "(None, 12, 12, 256)", "(None, 12, 12, 512)", "(None, 12, 12, 512)", 
        "(None, 6, 6, 512)", "(None, 6, 6, 512)", "(None, 18432)", "(None, 256)", 
        "(None, 256)", "(None, 256)", "(None, 7)"
    ],
    "Param #": [
        "5120", "2048", "1179904", "1024", 
        "0", "0", "295040", "512", 
        "147584", "512", "0", "0", 
        "295168", "1024", "1180160", "2048", 
        "0", "0", "0", "4718848", 
        "1024", "0", "1799"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(10, 15))

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create the table
table = ax.table(cellText=df.to_numpy(), colLabels=df.columns, cellLoc = 'left', loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Add header text
plt.title("CUDA is available. Configured TensorFlow to use the GPU.\nModel: 'sequential_22'\n", fontsize=12)

# Adding total parameters text below the table
plt.figtext(0.15, 0.02, "**Total Parameters:** 7,831,815 (29.88 MB)\n**Trainable Parameters:** 7,827,719 (29.86 MB)\n**Non-trainable Parameters:** 4,096 (16.00 KB)", ha="left", fontsize=10)

# Save the image
plt.savefig("../images/model_architecture.png", bbox_inches='tight')
plt.show()
