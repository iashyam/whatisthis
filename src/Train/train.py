import torch 
from tqdm import tqdm

class Trainer:

    def __init__(self, model: torch.nn.Module, criterion, optimizer, device):
        
        self.model = model
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.model = self.model.to(device)

    def training_step(self, epoch_no: int, dataloader)->tuple[float, float]:

        self.model.train()
        train_loss, train_accuracy = 0,0
        
        for batch, (x, y) in enumerate(bar := tqdm(dataloader)):
            
            x, y = x.to(self.device), y.to(self.device)

            y_pred = self.model(x)           #forward pass
            loss = self.criterion(y_pred, y) #loss calculation
            self.optimizer.zero_grad()     #zero grad
            loss.backward()                  #backward step
            self.optimizer.step()

            pred = y_pred.argmax(dim=0) 
            actual = y.argmax(dim=0) 
            accuracy = (actual==pred).float().mean()

            train_loss += loss
            train_accuracy += accuracy

            bar.set_description_str(f"epoch: {epoch_no}")
            bar.set_postfix_str(f"train_loss: {train_loss/(batch+1):4f}, train_accuracy: {train_accuracy/(batch+1):4f}")


        avg_loss = train_loss/len(dataloader)
        avg_accuracy = train_accuracy/len(dataloader)

        return avg_loss, avg_accuracy

    def test_step(self, epoch_no:int,  dataloader)->tuple[float, float]:

        self.model.eval()
        test_loss, test_accuracy = 0,0

        for batch, (x, y) in (bar := tqdm(enumerate(dataloader))):

            x, y = x.to(self.device), y.to(self.device)

            with torch.no_grad():

                y_pred = self.model(x)           #forward pass
                loss = self.criterion(y_pred, y)

                actual = y.argmax(dim=1) 
                pred = y_pred.argmax(dim=1) 
                accuracy = (actual==pred).float().mean()

                test_loss += loss
                test_accuracy += accuracy
                bar.set_description_str(f"epoch: {epoch_no}")
                bar.set_postfix_str(f"test_loss: {test_loss/(batch+1):4f}, test_accuracy: {test_accuracy/(batch+1):4f}")



        avg_loss = test_loss/len(dataloader)
        avg_accuracy = test_accuracy/len(dataloader)

        return avg_loss, avg_accuracy

    def train_loop(self, n_epochs: int, train_dataloader, test_dataloader) -> dict:

        history = {}
        history['train_loss'] = []
        history['test_loss'] = []
        history['train_accuracy'] = []
        history['test_accuracy'] = []

        for i in range(n_epochs):
            
            train_loss, train_accuracy = self.training_step(i, train_dataloader)
            test_loss, test_accuracy = self.test_step(i, test_dataloader)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_accuracy'].append(train_accuracy)
            history['test_accuracy'].append(test_accuracy)

        return history