import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from data import load_data
import utils
import line_auto_train
import conv_auto_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    "input_size": 784,
    "hidden_size": 121,
    "output_size": 10,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "weight_decay": 1e-5,
    "model_select": "all"
}
train_dataloader = load_data()
epochs = config["epochs"]
path = utils.create_result_folder()
writer = SummaryWriter(path[2])
auto_line_loss = line_auto_train.train(dataloader=train_dataloader,
                                       epochs=epochs,
                                       device=device,
                                       writer=writer)
auto_cov_loss = conv_auto_train.train(dataloader=train_dataloader,
                                      epochs=epochs,
                                      device=device,
                                      writer=writer)
print("train done!")
writer.close()
x = list(range(1, epochs + 1))
plt.plot(x, auto_line_loss, color='blue', label='auto_line_loss')
plt.plot(x, auto_cov_loss, color='red', label='auto_cov_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epoch for Different Models')
plt.show()
