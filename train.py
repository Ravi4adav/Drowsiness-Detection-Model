import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg19 import VGG19
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# To avoid overfitting due to image augmentation we are recommending creating dataframe which contains filepath and labels.
def extract_data_from_directory_to_df(path):
    df=pd.DataFrame({"Imagepath":[],"Label":[],"path":[],"Image":[]})
    for path,directory,files in os.walk(path):
        if len(files)>0:
            df['Image']=files
            df['path']=[path]*len(files)
            df['Imagepath']=df['path']+'/'+df['Image']
            df['Label']=[path.split('/')[-1]]*len(files)

    return df.drop(['path','Image'],axis=1)

temp1=extract_data_from_directory_to_df('./training_data/yawn')
temp2=extract_data_from_directory_to_df('./training_data/no_yawn')
temp3=extract_data_from_directory_to_df('./training_data/Open')
temp4=extract_data_from_directory_to_df('./training_data/Closed')
df=pd.concat([temp1,temp2, temp3, temp4],axis=0,ignore_index=True)

del temp1, temp2, temp3, temp4


# Splitting Training and test dataset
train, test= train_test_split(df, test_size=0.2, random_state=42,stratify=df['Label'])


# Create an instance of ImageDataGenerator for training and validation with a validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalizing pixel values between 0 and 1
    zoom_range=0.2,       # Applying zoom upto 20%
    rotation_range=25,
    horizontal_flip=True,  # Applying flipping horizontally at images
)

validation_datagen=ImageDataGenerator(rescale=1./255)



train_generator=train_datagen.flow_from_dataframe(train, x_col="Imagepath",
                                                 y_col="Label", target_size=(224,224),
                                                 batch_size=32,
                                                 class_mode='categorical')

validation_generator=validation_datagen.flow_from_dataframe(test,x_col='Imagepath',
                                                           y_col="Label",target_size=(224,224),
                                                           batch_size=32,
                                                           class_mode='categorical',shuffle=False)

# Training Model
model=Sequential()
# Layer-1 with 64 convolutions
model.add(Conv2D(64, (3,3), input_shape=(224,224,3), activation='relu'))
model.add(BatchNormalization())     # Applying Normalization to each batch for faster processing
model.add(MaxPooling2D())

# Layer-2 with 128 convolutions
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# Layer-2 with 128 convolutions
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# Layer-3 with 256 convolutions
model.add(Conv2D(256,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
# Applying Flatten layers
model.add(Flatten())


model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='softmax'))

# Defining Callback to stop early before execution of all epochs based on validation accuracy
early_stop=EarlyStopping(monitor='val_accuracy',patience=10,verbose=0,restore_best_weights=True)
# Defining Callback to reduce learning rate based on validation loss
learning_rate=ReduceLROnPlateau(monitor='val_accuracy',patient=2,verbose=1,factor=0.01)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=[early_stop,learning_rate])
# history=model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=[learning_rate])

model.save('/kaggle/working/drowsiness_model.h5')