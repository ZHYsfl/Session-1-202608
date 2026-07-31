import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from data import generate_samples



# 项目路径

PROJECT_PATH = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    PROJECT_PATH,
    "..",
    "data"
)



if __name__ == "__main__":


    print("Loading data...")


    csv_path = os.path.join(
        DATA_PATH,
        "driving_log.csv"
    )


    df = pd.read_csv(csv_path)



    print(
        "Dataset size:",
        len(df)
    )


    # 划分训练集

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )



    # CNN模型

    model = models.Sequential()


    model.add(
        Conv2D(
            16,
            (3,3),
            activation='relu',
            input_shape=(32,128,3)
        )
    )


    model.add(
        MaxPooling2D(
            pool_size=(2,2)
        )
    )


    model.add(
        Conv2D(
            32,
            (3,3),
            activation='relu'
        )
    )


    model.add(
        MaxPooling2D(
            pool_size=(2,2)
        )
    )


    model.add(
        Conv2D(
            64,
            (3,3),
            activation='relu'
        )
    )


    model.add(
        MaxPooling2D(
            pool_size=(2,2)
        )
    )


    model.add(
        Flatten()
    )


    model.add(
        Dense(
            500,
            activation='relu'
        )
    )


    model.add(
        Dropout(0.5)
    )


    model.add(
        Dense(
            100,
            activation='relu'
        )
    )


    model.add(
        Dropout(0.25)
    )


    model.add(
        Dense(
            20,
            activation='relu'
        )
    )


    model.add(
        Dense(1)
    )



    model.compile(
        optimizer=Adam(
            learning_rate=1e-4
        ),
        loss="mse"
    )



    model.summary()



    train_generator = generate_samples(
        train_df,
        DATA_PATH,
        augment=True
    )


    valid_generator = generate_samples(
        valid_df,
        DATA_PATH,
        augment=False
    )



    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_df)//128,
        epochs=30,
        validation_data=valid_generator,
        validation_steps=len(valid_df)//128
    )



    # 保存模型

    model.save(
        "model.h5"
    )


    print(
        "Training finished!"
    )


    # 保存loss

    import matplotlib.pyplot as plt


    plt.plot(
        history.history['loss'],
        label="train"
    )


    plt.plot(
        history.history['val_loss'],
        label="validation"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )


    plt.legend()


    plt.savefig(
        "../results/loss_curve.png"
    )


    plt.show()