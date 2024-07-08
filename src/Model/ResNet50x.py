from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

class ResNet50X():
    def __init__(self):
        self.model = self._create_model()

    def _create_model(self):
        resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in resnet50.layers:
            layer.trainable = False

        x = resnet50.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(136, activation='linear')(x)

        output = Reshape((68, 2))(x)

        model = Model(inputs=resnet50.input, outputs=output)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        return model

    def train_model(self, train_images, train_landmarks, test_image, test_landmarks):
        optimizer = Adam(learning_rate=0.001)
        # self.model.load_weights('/kaggle/input/filter-landmarks/keras/model1/1/model_checkpoint2.keras')
        checkpoint = ModelCheckpoint(
            filepath='model_checkpoint2.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        # # Train the model
        history = self.model.fit(train_images, train_landmarks, batch_size=32, epochs=5,
                            validation_data=(test_image, test_landmarks), callbacks=[checkpoint])
