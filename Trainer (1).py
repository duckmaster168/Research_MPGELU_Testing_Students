import torch

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 calculate_accuracy,
                 device: torch.device,
                 loss_steps: int = 1):
        
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.calculate_accuracy = calculate_accuracy
        self.device = device
        self.loss_steps = loss_steps

    def train(self, data_loader: torch.utils.data.DataLoader, epoch=None):
        train_loss, train_acc = 0.0, 0.0
        self.model.train()
        
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            
            train_loss += loss.item()
            train_acc += self.calculate_accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        
        grad_norms = self.model.get_layer_grad_norms() if hasattr(self.model, 'get_layer_grad_norms') else {}
        lambdas = self.model.get_lambda_values() if hasattr(self.model, 'get_lambda_values') else []

        if epoch is not None and epoch % self.loss_steps == 0:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%")

        return train_loss, train_acc, grad_norms, lambdas

    def test(self, data_loader: torch.utils.data.DataLoader, epoch=None):
        test_loss, test_acc = 0.0, 0.0
        self.model.eval()
        
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                test_pred = self.model(X)
                loss = self.loss_fn(test_pred, y)
                
                test_loss += loss.item()
                test_acc += self.calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            
            if epoch is not None and epoch % self.loss_steps == 0:
                print(f"Epoch {epoch:02d} | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")
                
            return test_loss, test_acc