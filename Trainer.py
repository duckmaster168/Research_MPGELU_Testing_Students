import torch

class Trainer:
    '''
    Trainer class handling training loops, evaluation steps, layer-wise gradient norm collection, 
    and sharpness parameter extraction[cite: 5].
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 calculate_accuracy,
                 device: torch.device,
                 loss_steps: int = 1):
        
        self.model = model[cite: 5]
        self.loss_fn = loss_fn[cite: 5]
        self.optimizer = optimizer[cite: 5]
        self.calculate_accuracy = calculate_accuracy[cite: 5]
        self.device = device[cite: 5]
        self.loss_steps = loss_steps[cite: 5]

    def train(self, data_loader: torch.utils.data.DataLoader, epoch: int = None):
        train_loss, train_acc = 0.0, 0.0[cite: 5]
        self.model.train()[cite: 5]
        
        batch_grad_norms = []

        for batch, (X, y) in enumerate(data_loader):[cite: 5]
            X, y = X.to(self.device), y.to(self.device)[cite: 5]
            y_pred = self.model(X)[cite: 5]
            loss = self.loss_fn(y_pred, y)[cite: 5]
            train_loss += loss.item()[cite: 5]
            train_acc += self.calculate_accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))[cite: 5]
            
            self.optimizer.zero_grad()[cite: 5]
            loss.backward()[cite: 5]

            # Record gradient norms across parameters
            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.detach().norm(2).item()
            batch_grad_norms.append(grad_norms)

            self.optimizer.step()[cite: 5]

        train_loss /= len(data_loader)[cite: 5]
        train_acc /= len(data_loader)[cite: 5]

        # Calculate average gradient norm for the epoch
        mean_grad_norms = {}
        if batch_grad_norms:
            keys = batch_grad_norms[0].keys()
            for k in keys:
                mean_grad_norms[k] = sum(b[k] for b in batch_grad_norms) / len(batch_grad_norms)

        # Extract sharpness parameters (lambda)
        lambdas = []
        for module in self.model.modules():
            if hasattr(module, 'lambda_param'):
                lambdas.append(module.lambda_param.detach().cpu().item())
            elif hasattr(module, 'get_lambda'):
                lambdas.append(module.get_lambda().detach().cpu().item())

        if epoch is not None and epoch % self.loss_steps == 0:[cite: 5]
            print(f"Epoch {epoch:02d} | Training Loss: {train_loss:.5f} | Training Accuracy: {train_acc:.2f}%")[cite: 5]
            
        return train_loss, train_acc, mean_grad_norms, lambdas

    def test(self, data_loader: torch.utils.data.DataLoader, epoch: int = None):
        test_loss, test_acc = 0.0, 0.0[cite: 5]
        self.model.to(self.device)[cite: 5]
        self.model.eval()[cite: 5]
        with torch.inference_mode():[cite: 5]
            for X, y in data_loader:[cite: 5]
                X, y = X.to(self.device), y.to(self.device)[cite: 5]
                test_pred = self.model(X)[cite: 5]
                loss = self.loss_fn(test_pred, y)[cite: 5]
                test_loss += loss.item()[cite: 5]
                test_acc += self.calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))[cite: 5]

            test_loss /= len(data_loader)[cite: 5]
            test_acc /= len(data_loader)[cite: 5]
            if epoch is not None and epoch % self.loss_steps == 0:[cite: 5]
                print(f"Epoch {epoch:02d} | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")[cite: 5]
            return test_loss, test_acc
