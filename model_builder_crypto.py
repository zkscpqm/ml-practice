from src.learning_04 import CryptoDataLoader, CryptoRNN


if __name__ == '__main__':
    modl = CryptoDataLoader()
    x1, y1, x2, y2 = modl.setup_training_data(percentage_separation=0.1, save=True, load_saved=True, ratio='LTC-USD')

    crypto_predict_rnn = CryptoRNN(tensor_board=True, model_checkpt=True)
    crypto_predict_rnn.build_and_validate_model(lstm_layer_sizes=(128, 128, 128),
                                                dense_layer_size=32,
                                                drouput_sizes=(.2, .1, .2, .2),
                                                train_x=x1, train_y=y1,
                                                test_x=x2, test_y=y2,
                                                batch_size=32,
                                                epochs=10)
