import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
import numpy as np
import os

print("TensorFlow version:", tf.__version__)

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

dataset = keras.preprocessing.image_dataset_from_directory(
    "PlantVillage", shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)

class_names = dataset.class_names
print(class_names)
print(len(dataset))

for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())

# take 1 is first batch
plt.figure(figsize=(10, 8))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    plt.show()
'''
# 80% is for training
# 20% out which 10% for validation and 10% for actual testing

# training size 80% is 0.8
train_size = 0.8
# len * train_size =  68*.08 = 54
train_ds = dataset.take(54)
print(len(train_ds))

# skip(54) is remaining 20%
test_ds = dataset.skip(54)

# will 10% for validation would 6

val_ds = test_ds.take(6)

test_ds = test_ds.skip(6)

print("Training dataset size ==", len(train_ds))
print("Validation dataset size ==", len(val_ds))
print("Actual testing dataset size ==", len(test_ds))
'''


def get_dataset_partition_tf(ds, train_split=0.8, val_split =0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    t_ds = ds.take(train_size)

    v_ds = ds.skip(train_size).take(val_size)
    tst_ds = ds.skip(train_size).skip(val_size)
    return t_ds, v_ds, tst_ds


train_ds, val_ds, test_ds = get_dataset_partition_tf(dataset)
print("Training dataset size ==", len(train_ds))
print("Validation dataset size ==", len(val_ds))
print("Actual testing dataset size ==", len(test_ds))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# now we need to scale and resize if not 256*256

resize_and_rescale = keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

# now we need to data augmentation to generate more training data set like rotated, contrast and zoomed images


data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
    # layers.experimental.preprocessing.RandomZoom(0.9)
])

# now create a model steps,
# build, compile, fit evaluate

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
classes_length = len(class_names)
model = keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(classes_length, activation="softmax")
])

# now need to build the model

model.build(input_shape=input_shape)

# now if we want we can print the summary
model.summary()

# next step is to compile the model with optimizers
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

# now we actually train out model with training dataset with fit method 50 epochs will be enough which take some time
historyOfEachEpoch = model.fit(train_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=val_ds)

print(historyOfEachEpoch.history.keys())
loss = historyOfEachEpoch.history['loss']
acc = historyOfEachEpoch.history['accuracy']
val_loss = historyOfEachEpoch.history['val_loss']
val_acc = historyOfEachEpoch.history['val_accuracy']


# accuracy chart
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc="lower right")
plt.title('Training and validation accuracy')
plt.show()


# loss chart
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc="upper right")
plt.title('Training and validation Loss')
plt.show()

# now we can test our model with actual test dataset
model.evaluate(test_ds)

# now we can check is model able to predict correctly

for image_batch, label_batch in test_ds.take(1):
    first_image = image_batch[0].numpy().astype('uint8')
    first_label = label_batch[0].numpy()

    print("Actual first Image to predict")
    plt.imshow(first_image)
    print("Actual Label: ", class_names[first_label])

    batch_predication = model.predict(image_batch)
    print("Predicated Label:", class_names[np.argmax(batch_predication[0])])
plt.show()

# now function which will take model and image as an input and predict the image


# def predict(keras_model, img):
#     img_array = keras.preprocessing.image.img_to_array(img[i].numpy())
#     img_array = tf.expand_dims(img_array, 0)
#
#     predictions = keras_model.predict(img_array)
#
#     predicated_class = class_names[np.argmax(predictions[0])]
#     confid = round(100 * (np.max(predictions[0])), 2)
#     return predicated_class, confid
#
#
# plt.figure(figsize=(15, 15))
# for imagess, labels in test_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#
#         predicted_class, confidence = predict(model, imagess[i], i)
#         actual_class = class_names[labels[i]]
#
#         ac_image = imagess[i].numpy().astype('uint8')
#         plt.imshow(ac_image)
#
#         # plt.title(f"Actual: {actual_class}, \n Predicted : {predicted_class}, \n Confidence: {confidence}")
#         plt.title(f"Actual: {actual_class}")
#         plt.axis("off")
#     plt.show()

model_version = max([int(i) for i in os.listdir("/Users/himanshusharma/PycharmProjects/potato-disease/models") + [0]])+1
model.save(f"/Users/himanshusharma/PycharmProjects/potato-disease/models/{model_version}")


