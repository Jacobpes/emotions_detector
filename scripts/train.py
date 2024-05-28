# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import RandomOverSampler
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Layer, Conv2D, Activation, Multiply, Add, BatchNormalization, GlobalMaxPooling2D, Reshape
# from tensorflow.keras.models import Model
# import matplotlib.pyplot as plt
# import face_recognition
# from imblearn.over_sampling import RandomOverSampler
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# # Define Channel and Spatial Attention Layers
# class ChannelAttention(Layer):
#     def __init__(self, ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.ratio = ratio

#     def build(self, input_shape):
#         self.shared_layer_one = Dense(input_shape[-1] // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
#         self.shared_layer_two = Dense(input_shape[-1], kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

#     def call(self, inputs):
#         avg_pool = GlobalAveragePooling2D()(inputs)
#         avg_pool = Reshape((1, 1, avg_pool.shape[-1]))(avg_pool)
#         avg_pool = self.shared_layer_one(avg_pool)
#         avg_pool = self.shared_layer_two(avg_pool)

#         max_pool = GlobalMaxPooling2D()(inputs)
#         max_pool = Reshape((1, 1, max_pool.shape[-1]))(max_pool)
#         max_pool = self.shared_layer_one(max_pool)
#         max_pool = self.shared_layer_two(max_pool)

#         scale = Activation('sigmoid')(Add()([avg_pool, max_pool]))
#         return Multiply()([inputs, scale])

# class SpatialAttention(Layer):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv2d = Conv2D(1, (kernel_size, kernel_size), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

#     def call(self, inputs):
#         avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
#         max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
#         concat = tf.concat([avg_pool, max_pool], axis=-1)
#         return Multiply()([inputs, self.conv2d(concat)])

# # Load data function
# def load_data_with_face_filter(filepath):
#     df = pd.read_csv(filepath)
#     df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))
#     X, y = [], []
#     for index, row in df.iterrows():
#         image = row['pixels'].reshape(48, 48).astype('uint8')
#         if face_recognition.face_locations(image, model="cnn") or face_recognition.face_locations(image, model="hog"):
#             X.append(row['pixels'] / 255.0)
#             y.append(row['emotion'])

#     print(f"Filtered out {len(df) - len(X)} images without faces")
#     X = np.array(X).reshape(-1, 48, 48, 1)
#     y = np.array(y)
#     ros = RandomOverSampler(random_state=40)
#     X_resampled, y_resampled = ros.fit_resample(X.reshape(-1, 48 * 48), y)
#     X_resampled = X_resampled.reshape(-1, 48, 48, 1)
#     y_resampled = to_categorical(y_resampled)
#     print(np.unique(y_resampled, return_counts=True))

#     return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=40)

# # Model building function with CBAM integration
# def build_model_with_cbam(input_shape=(48, 48, 1)):
#     base_model = ResNet50(include_top=False, weights=None, input_tensor=Input(shape=input_shape))
#     x = base_model.output

#     # Add CBAM after each convolution block in ResNet50
#     for layer in base_model.layers:
#         if isinstance(layer.output, tf.Tensor) and len(layer.output.shape) == 4:
#             # Apply CBAM modules to suitable layers
#             x = ChannelAttention()(x)
#             x = SpatialAttention()(x)

#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
#     x = Dropout(0.42)(x)
#     predictions = Dense(7, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def train_model(model, X_train, X_val, y_train, y_val):
#     if len(X_train) == 0 or len(y_train) == 0:
#         raise ValueError("Training dataset is empty after preprocessing and resampling.")
#     if len(X_val) == 0 or len(y_val) == 0:
#         raise ValueError("Validation dataset is empty after preprocessing and resampling.")
#     class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
#     class_weight_dict = dict(enumerate(class_weights))
#     callbacks = [
#         EarlyStopping(monitor='val_accuracy', patience=6, verbose=1, restore_best_weights=True),
#         ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=6, verbose=1)
#     ]
    
#     datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
#     print(X_train.shape, y_train.shape)
#     train_generator = datagen.flow(X_train, y_train, batch_size=64)
#     for data, labels in train_generator:
#         print(data.shape, labels.shape)
#         break  # Just check the first batch

#     history = model.fit(
#         train_generator,
#         epochs=100,
#         validation_data=(X_val, y_val),
#         callbacks=callbacks,
#         class_weight=class_weight_dict,
#         verbose=2
#     )
    
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.savefig('learning_curves.png')
#     plt.show()
#     return history


# def main():
#     X_train, X_val, y_train, y_val = load_data_with_face_filter('../data/train.csv')
#     model = build_model_with_cbam()
#     history = train_model(model, X_train, X_val, y_train, y_val)

# if __name__ == "__main__":
#     main()


import os
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D, Dropout, Layer, Conv2D, 
                                     Activation, Multiply, Add, BatchNormalization, GlobalMaxPooling2D, Reshape, MaxPooling2D, Flatten)
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
def load_data_with_face_filter(filepath):
    df = pd.read_csv(filepath)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, dtype=float, sep=' '))
    X, y = [], []
    for index, row in df.iterrows():
        image = row['pixels'].reshape(48, 48, 1)
        X.append(image)
        y.append(row['emotion'])
    X = np.array(X) / 255.0  # Normalize pixel values
    y = to_categorical(np.array(y))  # Convert labels to categorical format
    ros = RandomOverSampler(random_state=40)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(-1, 48 * 48), np.argmax(y, axis=1))
    X_resampled = X_resampled.reshape(-1, 48, 48, 1)
    y_resampled = to_categorical(y_resampled)
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=40)

# Define optimizers
optims = [
    Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
    Adam(learning_rate=0.001)
]

# Sequential model definition
model = Sequential([
    Conv2D(512, (5, 5), input_shape=(48, 48, 1), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_1'),
    BatchNormalization(name='batchnorm_1'),
    Conv2D(256, (5, 5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_2'),
    BatchNormalization(name='batchnorm_2'),
    MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'),
    Dropout(0.27, name='dropout_1'),
    Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_3'),
    BatchNormalization(name='batchnorm_3'),
    Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_4'),
    BatchNormalization(name='batchnorm_4'),
    MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'),
    Dropout(0.27, name='dropout_2'),
    Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_5'),
    BatchNormalization(name='batchnorm_5'),
    Conv2D(512, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_6'),
    BatchNormalization(name='batchnorm_6'),
    MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'),
    Dropout(0.27, name='dropout_3'),
    Flatten(name='flatten'),
    Dense(256, activation='elu', kernel_initializer='he_normal', name='dense_1'),
    BatchNormalization(name='batchnorm_7'),
    Dropout(0.27, name='dropout_4'),
    Dense(7, activation='softmax', name='out_layer')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.00005, patience=8, verbose=1, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=5, min_lr=1e-7, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
callbacks = [early_stopping, lr_scheduler, model_checkpoint]

def main():
    filepath = '/data/train.csv'
    if not os.path.exists(filepath):
        print(f"Error: The file {filepath} does not exist.")
        return
    X_train, X_val, y_train, y_val = load_data_with_face_filter(filepath)
    history = model.fit(X_train, y_train, epochs=60, validation_data=(X_val, y_val), batch_size=64, callbacks=callbacks)
    # save the model
    model.save('results/models/best_model.keras')

if __name__ == "__main__":
    main()

