from src.learning_02 import CatsVsDogsModeller

if __name__ == '__main__':

    dense_layer = tuple()
    conv_layer = (64, 128, 128)
    window_size = 3
    batch_size = 64
    val_split = 0.1

    name = '64-128-128-3-64-0.1-cnn-CVD-Kagg'
    modeller = CatsVsDogsModeller(size_x=80, tb=False, log_name=name)
    modeller.load_data(load_saved=True)
    modeller.build_and_validate_model(conv_layer_sizes=conv_layer,
                                      conv_window_size=window_size,
                                      dense_layers=dense_layer,
                                      batch_size=batch_size,
                                      val_split=val_split,
                                      epochs=10,
                                      savename='cnn-CVD-Kagg.model')
