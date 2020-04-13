from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def create_vgg_based_model(input_image_size, n_output_classes,
                           vgg_type=None, vgg_layers_to_train=None):
    """
    Creates and returns a NN based on VGG backbone.
    Model = Base Model (VGG) + Extra layers to train
    
    """
    if vgg_type not in ['VGG16', 'VGG19']:
      raise Exception('vgg_type has an invalid value')

    # backbone model
    if vgg_type == 'VGG16':
      base_model = VGG16(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(input_image_size,
                                                   input_image_size,
                                                   3)))
    else:
      base_model = VGG19(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(input_image_size,
                                                   input_image_size,
                                                   3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(64, activation="relu")(head_model)
    # head_model = Dropout(0.5)(head_model)
    head_model = Dense(n_output_classes, activation="softmax")(head_model)
    
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in base_model.layers:
        if layer.name in vgg_layers_to_train:
            layer.trainable = True
        else:
            layer.trainable = False
    
    return model