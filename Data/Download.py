import requests

# download data from the link


def Download_data(link: str)->None:
    requests.get(link)

if __name__=="__main__":

    link = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    Download_data(link)