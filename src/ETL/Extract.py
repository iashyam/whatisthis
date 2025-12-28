from pathlib import Path
import numpy as np
import PIL
import os

class Extractor:
    def __init__(self, file: Path):

        if not file.exists():
            raise FileExistsError
        
        self.file = file

    def unpickle(self):
        import pickle
        with open(self.file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        print(dict.keys())
        return dict

    something = "this is somethingk"

    def extract(self):
        dict = self.unpickle()
        datas = np.asarray(dict[b"data"], dtype=np.uint8).astype(int)
        labels = np.asarray(dict[b"labels"], dtype=np.uint8).astype(int)
        print(len(datas))
        datas = datas.reshape(10000, 3, 32, 32)

        return datas, labels

     

if __name__=="__main__":
    print(os.getcwd())
    datas, labels = Extractor(Path("Data/cifar-10-batches-py/data_batch_4")).extract()
    i = np.random.randint(datas.shape[0])
    image = datas[i]
    image = image.reshape(32, 32, 3)
    print(image.shape)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title(labels[i])
    plt.axis("off")
    plt.show() 
