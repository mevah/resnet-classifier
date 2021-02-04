import os

import click
import keras
import numpy as np
import matplotlib.pyplot as plt 

def main():
    resnet50 = keras.models.load_model('models/resnet50.h5')
    make_prediction(resnet50)


"""
Make Prediction [using pre-trained model]
"""

def make_prediction(model):


    categories = os.listdir('/itet-stor/himeva/net_scratch/resnet_data/test/')

    file_path = "/itet-stor/himeva/net_scratch/resnet_data/test/"
    normal_imgs= os.listdir(file_path + "normal/")
    lesion_imgs = os.listdir(file_path + "lesion/")
    accs= []
    for imgg in normal_imgs:

        # preprocessing
        img = keras.preprocessing.image.load_img(file_path + "normal/"+ imgg, target_size=(64, 64))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.utils.normalize(x)

        # make prediction
        preds = model.predict(x)
        print(preds)
        
        #print("Model predicts a \"{}\" with {:.2f}% probability".format(
        #categories[np.argmax(preds[0])], preds[0][np.argmax(preds)] * 100))
        prediction = np.argmax(preds[0])
        if prediction == 0:
            accs.append(0)
            print("wrong prediction for file: ", imgg)

        else:
            accs.append(1)
    
    for imgg in lesion_imgs:

        # preprocessing
        img = keras.preprocessing.image.load_img(file_path + "lesion/"+ imgg, target_size=(64, 64))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.utils.normalize(x)

        # make prediction
        preds = model.predict(x)
        #print("Model predicts a \"{}\" with {:.2f}% probability".format(
        #categories[np.argmax(preds[0])], preds[0][np.argmax(preds)] * 100))
        prediction = np.argmax(preds[0])
        if prediction == 0:
            accs.append(1)

        else:
            accs.append(0)
            print(preds)
            print("wrong prediction for file: ", imgg)

    

    print("Mean accuracy: ",np.mean(accs))
    print("There were " + str(len(accs)) + "images.")
if __name__ == '__main__':
    main()
