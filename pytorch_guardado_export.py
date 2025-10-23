import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import onnxruntime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# preparamos los datos

dataloader = {
    'train': torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ])
                      ), batch_size=2048, shuffle=True, pin_memory=True),
    'test': torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=False,
                   transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                        ])
                     ), batch_size=2048, shuffle=False, pin_memory=True)
}


# definimos el modelo

def block(c_in, c_out, k=3, p=1, s=1, pk=2, ps=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(pk, stride=ps)
    )

class CNN(torch.nn.Module):
  def __init__(self, n_channels=1, n_outputs=10):
    super().__init__()
    self.conv1 = block(n_channels, 64)
    self.conv2 = block(64, 128)
    self.fc = torch.nn.Linear(128*7*7, n_outputs)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x
  



# entrenamos el modelo

def fit(model, dataloader, epochs=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader['train'])
        for batch in bar:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_acc.append(acc)
            bar.set_description(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")
        bar = tqdm(dataloader['test'])
        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                val_acc.append(acc)
                bar.set_description(f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}")
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f} acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f}")


model = CNN()
fit(model, dataloader)



# guardar modelo

PATH = './checkpoint.pt'
torch.save(model.state_dict(), PATH)


# cargar modelo

state_dict = torch.load(PATH, map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(state_dict)
model.eval()



def evaluate(model, dataloader):
    model.eval()
    model.to(device)
    bar = tqdm(dataloader['test'])
    acc = []
    with torch.no_grad():
        for batch in bar:
            X, y = batch
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            acc.append((y == torch.argmax(y_hat, axis=1)).sum().item() / len(y))
            bar.set_description(f"acc {np.mean(acc):.5f}")


evaluate(model, dataloader)


PATH1 = './model.pt'
torch.save(model, 'model.pt')

model = torch.load(PATH1, map_location=torch.device('cpu'), weights_only=False)
model.eval()





# tracing

x = torch.rand(32, 1, 28, 28)
traced_model = torch.jit.trace(model.cpu(), x)
traced_model.save('model.zip')

loaded_model = torch.jit.load('model.zip')
evaluate(loaded_model, dataloader)



# scripting

scripted_model = torch.jit.script(model.cpu())
scripted_model.save('model.zip')

loaded_model = torch.jit.load('model.zip')
evaluate(loaded_model, dataloader)



# ---- Onnx ----

x = torch.rand(32, 1, 28, 28)
y = model.cpu()(x)

# exportamos el modelo
torch.onnx.export(model,                     # el modelo
                  x,                         # un ejemplo del input
                  "model.onnx",              # el nombre del archivo para guardar el modelo
                  export_params=True,        # guardar los pesos de la red
                  opset_version=10,          # versión de ONNX
                  do_constant_folding=True,  # optimizaciones
                  input_names = ['input'],   # nombre de los inputs
                  output_names = ['output'], # nombre de los outputs
                  dynamic_axes={'input' : {0 : 'batch_size'},    # ejes con longitud variable (para poder usar diferentes tamaños de batch)
                                'output' : {0 : 'batch_size'}})





def onnx_evaluate(model, dataloader):
    # cargarmos el modelo
    ort_session = onnxruntime.InferenceSession(model)
    bar = tqdm(dataloader['test'])
    acc = []
    with torch.no_grad():
        for batch in bar:
            X, y = batch
            X, y = X.numpy(), y.numpy()
            # generamos los inputs
            ort_inputs = {ort_session.get_inputs()[0].name: X}
            # extraemos los outputs
            ort_outs = ort_session.run(None, ort_inputs)[0]
            acc.append((y == np.argmax(ort_outs, axis=1)).mean())
            bar.set_description(f"acc {np.mean(acc):.5f}")


onnx_evaluate("model.onnx", dataloader)



class Preprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # esperamos un batch de imágenes sin normalizar
        # normalización
        x = (x / 255.)
        x = (x - 0.1307) / 0.3081
        # dimsensiones -> [bs, c, h, w]
        x = x.unsqueeze(1)
        # en imágenes en color, haríamos un `permute`
        return x

class Postprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x) :
        # devolvemos distribución de probabilidad
        # y clase con mayor probabilidad
        return self.softmax(x), torch.argmax(x, dim=1)


final_model = torch.nn.Sequential(
    Preprocessing(),
    model.cpu(),
    Postprocessing()
)

scripted_model = torch.jit.script(final_model)
scripted_model.save('model.zip')