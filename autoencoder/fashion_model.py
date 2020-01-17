import keras
import files

class FashionModel():
    
    def __init__(self, shape_img, n_classes):
        
        self.shape_img = shape_img
        self.n_classes = n_classes
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.classifier = None
        self.losses = ['binary_crossentropy', 'categorical_crossentropy']
        self.optimizer = 'adam'
        self.metrics = ['accuracy']

    def build(self):

        n_hidden_1, n_hidden_2, n_hidden_3 = 16, 8, 8
        convkernel = (3, 3)  # convolution kernel
        poolkernel = (2, 2)  # pooling kernel

        input_layer = keras.layers.Input(shape=self.shape_img)
        x = keras.layers.Conv2D(n_hidden_1, convkernel, activation='relu', padding='same')(input_layer)
        x = keras.layers.MaxPooling2D(poolkernel, padding='same')(x)
        x = keras.layers.Conv2D(n_hidden_2, convkernel, activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(poolkernel, padding='same')(x)
        x = keras.layers.Conv2D(n_hidden_3, convkernel, activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D(poolkernel, padding='same', name='output_encoder')(x)

        x = keras.layers.Conv2D(n_hidden_3, convkernel, activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D(poolkernel)(x)
        x = keras.layers.Conv2D(n_hidden_2, convkernel, activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D(poolkernel)(x)
        x = keras.layers.Conv2D(n_hidden_1, convkernel, activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D(poolkernel)(x)
        decoded = keras.layers.Conv2D(self.shape_img[2], convkernel, activation='sigmoid', padding='same', name='output_decoder')(x)

        classification = keras.layers.Flatten()(encoded)
        classification = keras.layers.Dense(self.n_classes, activation='softmax', name='output_classifier')(classification)

        # Create autoencoder model
        autoencoder = keras.Model(input_layer, [decoded, classification])
        input_autoencoder_shape = autoencoder.layers[0].input_shape[1:]
        output_autoencoder_shape = autoencoder.layers[-1].output_shape[1:]

        # Create encoder model
        encoder = keras.Model(input_layer, encoded)
        input_encoder_shape = encoder.layers[0].input_shape[1:]
        output_encoder_shape = encoder.layers[-1].output_shape[1:]

        # Create decoder model
        decoded_input = keras.Input(shape=output_encoder_shape)
        decoded_output = autoencoder.layers[-9](decoded_input)  # Conv2D
        decoded_output = autoencoder.layers[-8](decoded_output)  # UpSampling2D
        decoded_output = autoencoder.layers[-7](decoded_output)  # Conv2D
        decoded_output = autoencoder.layers[-6](decoded_output)  # UpSampling2D
        decoded_output = autoencoder.layers[-5](decoded_output)  # Conv2D
        decoded_output = autoencoder.layers[-4](decoded_output)  # UpSampling2D
        decoded_output = autoencoder.layers[-3](decoded_output)  # Conv2D

        decoder = keras.Model(decoded_input, decoded_output)

        # Create classifier model
        classifier = keras.Model(input_layer, classification)

        # Assign models
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def compile(self):

        self.autoencoder.compile(optimizer=self.optimizer, 
            loss=self.losses, 
            metrics=self.metrics)
