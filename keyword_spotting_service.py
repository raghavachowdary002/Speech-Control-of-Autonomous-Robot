import librosa
import tensorflow as tf
import numpy as np
import sounddevice as sd
import soundfile as sf

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050


class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    _mapping = [
        "Go_To_Home_Position",
        "Go_To_Kitchen",
        "Go_To_Living_Room",
        "Go_To_Origin",
        "Go_To_Room_One",
        "Go_To_Room_Two",
        "Move_To_Home_Position",
        "Move_To_Kitchen",
        "Move_To_Living_Room",
        "Move_To_Origin",
        "Move_To_Room_One",
        "Move_To_Room_Two"
    ]
    _instance = None

    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    # duration=2
    # filename = 'GO_TO_ORIGIN.wav'
    # print("about to start")
    # print("start")
    # mydata = sd.rec(int(SAMPLES_TO_CONSIDER * duration), samplerate=SAMPLES_TO_CONSIDER,
    #             channels=1, blocking=True)
    # print("end")
    # sd.wait()
    # sf.write(filename, mydata, SAMPLES_TO_CONSIDER)
    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    # kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    # assert kss is kss1

    # make a prediction
    #already trained file(correct prediction)
    keyword1 = kss.predict("test/move2room2.wav")
    #already trained file(correct prediction)
    keyword2 = kss.predict("test/go2room2.wav")
    #new file(correct prediction)
    keyword3=kss.predict("test/mov2room2.wav")
    #new file(wrong prediction)
    keyword4=kss.predict("test/mroom2.wav")
    #already trained file(correct prediction)
    keyword8=kss.predict("test/12.wav")


    print(keyword1)
    print(keyword2)
    print(keyword3)
    print(keyword4)
    print(keyword8)

