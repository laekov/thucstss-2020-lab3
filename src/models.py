from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape, Concatenate
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D


from specs import WIDTH, HEIGHT, CHANNEL


class BaseCNNModel(Model):
    def __init__(self):
        super(BaseCNNModel, self).__init__()
        self.encoder = Sequential([
            Conv2D(32, (5, 5), padding='valid', 
                input_shape=(HEIGHT, WIDTH, CHANNEL), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),

            # TODO for task #1: Add a convolution layer here
            # TODO for task #1: Add another pooling layer here

            Dropout(0.15),
            Conv2D(32, (3, 3), activation='relu'),
            AveragePooling2D(pool_size=(2, 2)),
            Dropout(0.15),
            Flatten()
        ])

        self.decoders = [Dense(10, activation='softmax') for _ in range(4)]
        self.concat = Concatenate()
        self.reshape = Reshape((4, 10))


    def call(self, x):
        x = self.encoder(x)
        x = self.concat([decoder(x) for decoder in self.decoders])
        x = self.reshape(x)
        return x


class AdvancedModel(Model):
    # TODO for task #2: Create your own model and pursue higher accuracy!
    pass
