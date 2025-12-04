from pathlib import Path
import numpy as np
import PIL

class Extractor:
    def __init__(self, file: Path):

        if not file.exists():
            raise FileExistsError
        
        self.file = file

    def unpickle(self):
        import pickle
        with open(self.file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def extract(self):
        dict = self.unpickle()
        datas = np.asarray(dict[b"data"], dtype=np.uint8).astype(int)
        labels = np.asarray(dict[b"fine_labels"], dtype=np.uint8).astype(int)
        datas = datas.reshape(10000, 3, 32, 32)
        datas = np.transpose(datas, (0, 2, 3, 1))  # H W C

        return datas, labels

     

if __name__=="__main__":
    datas, labels = Extractor(Path("cifar-100-python/test")).extract()
    i = np.random.randint(datas.shape[0])
    image = datas[i]
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title(labels[i])
    plt.axis("off")
    plt.show() 