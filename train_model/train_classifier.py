"""
Training image classifier.

From TensorFlow tutorial:
https://www.tensorflow.org/tutorials/images/classification
"""


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def main():
    batch_size = 32
    img_width = 200
    img_height = 450
    data_dir = "Labeled_Frames/"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(
            img_height, img_width, 3)),
        layers.Conv2D(16, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 5, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # Train model
    epochs = 3
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save the entire model
    model.save('Model/last_trained_model')


if __name__ == '__main__':
    main()

