import os
import click
import logging

import keras
import numpy as np
import keras.backend as K

from model import create_model
import wandb
from wandb.keras import WandbCallback

K.set_image_data_format('channels_last')

"""
Train Model [optional args]
"""
@click.command(name='Training Configuration')
@click.option(
    '-n',
    '--name',
    default="model",
    help='model name to create folders with'
)
@click.option(
    '-lr', 
    '--learning-rate', 
    default=0.005, 
    help='Learning rate for minimizing loss during training'
)
@click.option(
    '-bz',
    '--batch-size',
    default=32,
    help='Batch size of minibatches to use during training'
)
@click.option(
    '-ne', 
    '--num-epochs', 
    default=50, 
    help='Number of epochs for training model'
)
@click.option(
    '-se',
    '--save-every',
    default=1,
    help='Epoch interval to save model checkpoints during training'
)
@click.option(
    '-tb',
    '--tensorboard-vis',
    is_flag=True,
    help='Flag for TensorBoard Visualization'
)
@click.option(
    '-ps',
    '--print-summary',
    is_flag=True,
    help='Flag for printing summary of the model'
)

def train(name, learning_rate, batch_size, num_epochs, save_every, tensorboard_vis, print_summary):

    # Set an experiment name to group training and evaluation
    experiment_name = name

    # Start a run, tracking hyperparameters
    wandb.init(
        project="resnet50",
        group=experiment_name,
        config={

            "optimizer":"adam",
            "loss": "custom_loss",
            "metric": ["accuracy", "precision"],
            "epoch": num_epochs,
            "batch_size": batch_size
        })
    config = wandb.config
    setup_paths(name)

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)

    get_gen = lambda x: datagen.flow_from_directory(
        '/itet-stor/himeva/net_scratch/final_data/fullres/{}'.format(x),
        target_size=(320, 320),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    #save_to_dir="/itet-stor/himeva/net_scratch/aug_images/{}".format(x)
    )

    # generator objects
    train_generator = get_gen('train')
    val_generator = get_gen('val')
    test_generator = get_gen('test')

    if os.path.exists('models/' + name + '/resnet50.h5'):
        # load model
        logging.info('loading pre-trained model')
        resnet50 = keras.models.load_model('models/' + name + '/resnet50.h5')
    else:
        # create model
        logging.info('creating model')
        resnet50 = create_model(input_shape=(320, 320, 1), classes=3)
    
    #optimizer = keras.optimizers.Adam(learning_rate)
    
    #change these into more useful ometrics and losses
    optimizer = keras.optimizers.Adam(learning_rate)
    resnet50.compile(optimizer=optimizer, loss=custom_loss(),
                     metrics=['accuracy', keras.metrics.Precision()])
    
    if print_summary:
        resnet50.summary()

    callbacks = configure_callbacks(save_every, tensorboard_vis)

    # train model
    logging.info('training model')
    resnet50.fit_generator(
        train_generator,
        steps_per_epoch=6000//config.batch_size,
        epochs=config.epoch,
        verbose=1,
        validation_data=val_generator,
        validation_steps=1000//config.batch_size,
        shuffle=True,
        callbacks=callbacks
    )
    # save model
    logging.info('Saving trained model to models/' + name + '/resnet50.h5')
    resnet50.save('models/' + name + '/resnet50.h5')

    # evaluate model
    logging.info('evaluating model')
    preds = resnet50.evaluate_generator(
        test_generator,
        steps=1000//config.batch_size,
        verbose=1
    )
    logging.info('test loss: {:.4f} - test acc: {:.4f}'.format(preds[0], preds[1]))

    wandb.finish()
    #keras.utils.plot_model(resnet50, to_file='models/fold2resnet50.png')

"""
Configure Callbacks for Training
"""
 
# implement mean squared false error
def custom_loss():
    def loss(y_true,y_pred):    
        print(y_true.shape, y_pred.shape)
        neg_y_true = 1 - y_true
        #neg_y_pred = 1 - y_pred
        fp = (neg_y_true * y_pred)
        #tn = K.sum(neg_y_true * neg_y_pred)
        #fn = K.sum(neg_y_pred * y_true)
        num = K.sum(fp, axis = -1)
        den= K.sum(y_true, axis= -1)

        # Converted as Keras Tensors

        #specificity = TN / (TN + FP + K.epsilon())
        #recall = TP / (TP + FN + K.epsilon())

        return K.square(num/den) 
    return loss

def configure_callbacks(save_every=1, tensorboard_vis=False):
    # checkpoint models only when `val_loss` impoves
    saver = keras.callbacks.ModelCheckpoint(
        'models/ckpts/fresmodel.ckpt',
        monitor='val_loss',
        save_best_only=True,
        period=save_every,
        verbose=1
    )
    
    # reduce LR when `val_loss` plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        min_lr=1e-10
    )

    # early stopping when `val_loss` stops improving
    early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=15, 
        verbose=1
    )

    callbacks = [saver, reduce_lr, early_stopper]

    if tensorboard_vis:
        # tensorboard visualization callback
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_cb)
    
    callbacks.append(WandbCallback())
    return callbacks

def setup_paths(name):
    if not os.path.isdir("models/"+name+"/ckpts"):
        if not os.path.isdir('models'):
            os.mkdir('models')
        os.mkdir("models/"+name)
        os.mkdir("models/"+name+"/ckpts")

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(
        format=LOG_FORMAT, 
        level='INFO'
    )

    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()
