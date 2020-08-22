
from src.learning_02 import CatsVsDogsModeller

if __name__ == '__main__':

    modeller = CatsVsDogsModeller(size_x=80, savename='cnn-CVD-Kagg.model')
    while True:
        fn = input('Input filename: ')
        prediction = modeller.predict_image(filename=fn)
        print(prediction)


