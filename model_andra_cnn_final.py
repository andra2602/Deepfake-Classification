import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# Setarile de baza
# Se stabileste daca modelul va rula pe GPU sau CPU -> in cazul meu rularea a avut loc pe CPU
dispozitivAAM = "cuda" if torch.cuda.is_available() else "cpu"
DIMENSIUNE_batch = 64
NUMAR_clase = 5  # conform cerintei competitiei -> 5 categorii de clasare a imaginilor

# ------------------------------------
# Aici are loc preprocesarea imaginilor
# Acestea sunt redimensionate 128x128, pentru a se pleca de la aceeasi forma
# Augumentate si normalizate [-1,1]
# ------------------------------------
img_preprocesare = transforms.Compose([
    transforms.Resize((128, 128)),
    # imaginea este oglindita stanga/dreapta cu o probabilitate de 50%
    transforms.RandomHorizontalFlip(p=0.5), 
    # imaginea este rotita cu un unghi in intervalul [-15,15] grade
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # imaginea e normalizata intre [-1, 1] pe toate cele 3 canale RGB
])


# Aici imi definesc o clasa dataset personalizata
class AndraDfDataset(Dataset):
    def __init__(self, cale_fisierCSV, cale_imagini, img_preprocesare=None):
        self.tabelDATE = pd.read_csv(cale_fisierCSV)
        self.cale_imagini = cale_imagini
        self.img_preprocesare = img_preprocesare

    def __len__(self):
        return len(self.tabelDATE)

    def __getitem__(self, idx):
        img_id = self.tabelDATE.iloc[idx, 0]
        # dupa ce este preluat ID-ul imaginii, se construieste calea catre aceasta cu extensia .png ,mai apoi imaginea este incarcata si convertita in format RGB
        caleCompletaImagine = os.path.join(self.cale_imagini, f"{img_id}.png")
        inputImagine = Image.open(caleCompletaImagine).convert("RGB")

        label = int(self.tabelDATE.iloc[idx, 1])
        if self.img_preprocesare: #se aplica preprocesarea imaginilor definita mai sus
            inputImagine = self.img_preprocesare(inputImagine)

        return inputImagine, label
    

# Arhitectura CNN
class AndraModelCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------Definirea straturilor convolutionale----------------------------------------
        # Observatie: parametrul BatchNorm2d trebuie sa fie *egal cu numarul de filtre*
        # din stratul convolutional anterior, deoarece normalizeaza fiecare canal de iesire
        # independent (adica fiecare harta de activare).
        # Primul strat
        self.primul_layerCONV = nn.Sequential(
            # Preia imaginea RGB (3 canale de culoare), aplica 16 filtre 3x3 si normalizeaza activarile
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # BatchNorm2d stabilizeaza invatarea,scade riscul de overfitting si normalizeaza valorile activarii
            nn.ReLU() # functia de activare --> elimina valorile negative
        )
        # Al doilea strat
        self.alDoilea_layerCONV = nn.Sequential(
            # Similar cu primul strat, dar continua procesarea , primeste cele 16 harti de activare, aplica 32 de filtre 3x3, pastrand dimensiunea spatiala
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # BatchNorm2d stabilizeaza invatarea,scade riscul de overfitting si normalizeaza valorile activarii
            nn.ReLU()# functia de activare --> elimina valorile negative
        )
        # Al treilea strat
        self.alTreilea_layerCONV = nn.Sequential(
            # Similar cu al doilea strat, dar continua procesarea , primeste cele 32 harti de activare, aplica 64 de filtre 3x3, pastrand dimensiunea spatiala
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # BatchNorm2d stabilizeaza invatarea,scade riscul de overfitting si normalizeaza valorile activarii
            nn.ReLU()# functia de activare --> elimina valorile negative
        )
        # Al patrulea strat
        self.alPatrulea_layerCONV = nn.Sequential(
            # Similar cu al treilea strat, dar continua procesarea , primeste cele 64 harti de activare, aplica 128 de filtre 3x3, pastrand dimensiunea spatiala
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # BatchNorm2d stabilizeaza invatarea,scade riscul de overfitting si normalizeaza valorile activarii
            nn.ReLU()# functia de activare --> elimina valorile negative
        )

        # Pool-ul reduce dimensiunele la jumatate dupa fiecare bloc convolutional
        # Acesta ajuta la prevenirea overfitting-ului si extragerea caracteristicilor dominante
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Definesc acest dropout pentru a preveni sansa de overfitting
        # Astfel 30% din neuroni sunt dezactivati in mod aleator
        self.dropout = nn.Dropout(0.3)# am ales 0.3 deoarece este l-am considerat a fi un dropout echilibrat

        # Intrarea este facuta 1-dimensionala si redusa la 256 de neuroni
        self.primul_layerFullyConnected = nn.Linear(128 * 8 * 8, 256)
        # cei 256 de neuroni sunt mapati la numarul de clase
        self.alDoilea_layerFullyConnected = nn.Linear(256, NUMAR_clase) 

    def forward(self, CNN_AAM_X):
        # In aceste linii imaginea intra pe rand in fiecare bloc conv->BatchNorm-> ReLu apoi se face pooling
        CNN_AAM_X = self.pool(self.primul_layerCONV(CNN_AAM_X)) # 16x64x64
        CNN_AAM_X = self.pool(self.alDoilea_layerCONV(CNN_AAM_X)) # 32x32x32
        CNN_AAM_X = self.pool(self.alTreilea_layerCONV(CNN_AAM_X)) # 64x16x16
        CNN_AAM_X = self.pool(self.alPatrulea_layerCONV(CNN_AAM_X)) # 128x8x8
        # Aplatizare + clasificare prin fully connected layers
        CNN_AAM_X = CNN_AAM_X.view(CNN_AAM_X.size(0), -1) 
        # Dropout-ul este aplicat inainte si dupa primul fully connected layer pentru regularizare
        CNN_AAM_X = self.dropout(CNN_AAM_X)
        CNN_AAM_X = F.relu(self.primul_layerFullyConnected(CNN_AAM_X))
        CNN_AAM_X = self.dropout(CNN_AAM_X)
        # Stratul final face mapping-ul catre cele 5 clase de iesire
        CNN_AAM_X = self.alDoilea_layerFullyConnected(CNN_AAM_X)
        return CNN_AAM_X


# Se initializeaza atat setul de antrenare, cat si cel de validare 
datePentruAntrenare = AndraDfDataset(
    cale_fisierCSV="train.csv",
    cale_imagini="train",
    img_preprocesare=img_preprocesare
)

datePentruValidare = AndraDfDataset(
    cale_fisierCSV="validation.csv",
    cale_imagini="validation",
    img_preprocesare=img_preprocesare
)
# Loadere pentru train si validare cu shuffle activ doar la antrenare pentru diversificare
loaderPentruAntrenare = DataLoader(datePentruAntrenare, batch_size = DIMENSIUNE_batch, shuffle = True)
loaderPentruValidare = DataLoader(datePentruValidare, batch_size = DIMENSIUNE_batch, shuffle = False)

funcPierdere = nn.CrossEntropyLoss() # Functia de pierdere - CrossEntropyLoss pentru ca este potrivita pentru clasificare in clase multiple
model = AndraModelCNN().to(dispozitivAAM)
# Optimizatorul folosit este SGD, cu un learning rate fix si momentum
# Momentum accelereaza antrenarea si ajuta la evitarea stagnarii în minime locale
AAMoptimizator = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(AAMoptimizator, step_size=7, gamma=0.5)# Scheduler-ul scade learning rate-ul la fiecare 7 epoci, cu un factor de 0.5, il folosesc pentru a stabiliza invatarea

NUMAR_epoci = 60
for epocaCurenta in range(NUMAR_epoci):
    model.train()# se seteaza modelul in modul de antrenare
    running_loss = 0.0
     # Pentru fiecare batch din setul de antrenare, mut imaginile si etichetele acestora pe CPU/GPU -> in cazul meu pe CPU
    for batchIDX, (imgIncarcatorAntrenare, labelsIncarcatorAntrenare) in enumerate(loaderPentruAntrenare):
        imgIncarcatorAntrenare, labelsIncarcatorAntrenare = imgIncarcatorAntrenare.to(dispozitivAAM), labelsIncarcatorAntrenare.to(dispozitivAAM)
        AAMoptimizator.zero_grad()
        # obtin predictia modelului si calculez pierderea
        predictii = model(imgIncarcatorAntrenare)
        loss = funcPierdere(predictii, labelsIncarcatorAntrenare)
        # se aplica backpropagation : pentru actualizarea ponderilor modelului
        loss.backward()
        AAMoptimizator.step()

        running_loss += loss.item()
        # afisare progres in consola la fiecare 100 batch-uri
        if batchIDX % 100 == 0:
             print(f"[Pentru epoca {epocaCurenta+1}], cu batch-ul: {batchIDX}, pierderea curenta este: {loss.item():.4f}")
    scheduler.step() # se ajusteaza learning rate-ul conform scheduler-ului
    valoareLRdupaEpoca = scheduler.get_last_lr()[0]
    print(f"Rata de invatare dupa epoca {epocaCurenta+1} este: {valoareLRdupaEpoca:.6f}")


# Aici are loc evaluarea pe setul de validare
model.eval()  # se trece in modul de evaluare
correct = 0
total = 0
total_loss = 0

with torch.no_grad():  # In timpul evaluarii, se dezactiveaza calculul de gradient pentru a economisi resurse
    for imgIncarcatorAntrenare, labelsIncarcatorAntrenare in loaderPentruValidare:
        imgIncarcatorAntrenare, labelsIncarcatorAntrenare = imgIncarcatorAntrenare.to(dispozitivAAM), labelsIncarcatorAntrenare.to(dispozitivAAM)
        iesiriModel = model(imgIncarcatorAntrenare)
        # calcularea pierderii
        loss = funcPierdere(iesiriModel, labelsIncarcatorAntrenare)
        total_loss += loss.item()
        
        _, etichetePrezise = torch.max(iesiriModel, 1)
        correct += (etichetePrezise == labelsIncarcatorAntrenare).sum().item()# vedem cate predictii au fost corecte
        total += labelsIncarcatorAntrenare.size(0)

# Se calculeaza si afiseaza acuratetea si pierderea medie
acurateteValidare = 100 * correct / total
lossMediu = total_loss / len(loaderPentruValidare)

print(f"\nPerformanta modelului MLP pe setul de validare: {acurateteValidare:.2f}%")
print(f"Loss-ul mediu la validare: {lossMediu:.4f}")

# Matricea de confuzie pentru modelul CNN

preziceriMatriceConfuzie = []
labelsMatriceConfuzie = []

with torch.no_grad():
    for imgIncarcatorAntrenare, labelsIncarcatorAntrenare in loaderPentruValidare:
        imgIncarcatorAntrenare = imgIncarcatorAntrenare.to(dispozitivAAM)
        iesiriModel = model(imgIncarcatorAntrenare)
        # Alegem clasa cu cea mai mare probabilitate 
        _, etichetePrezise = torch.max(iesiriModel, 1)
         # prediciile (convertite in numpy array) sunt salvate pentru analiza finala
        preziceriMatriceConfuzie.extend(etichetePrezise.cpu().numpy())
        labelsMatriceConfuzie.extend(labelsIncarcatorAntrenare.numpy())
# matricea de confuzie este construita comparand valorile reale cu cele prezise
matriceConfuzieCNN = confusion_matrix(labelsMatriceConfuzie, preziceriMatriceConfuzie)
disp = ConfusionMatrixDisplay(confusion_matrix=matriceConfuzieCNN, display_labels=[0, 1, 2, 3, 4])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de Confuzie - CNN")
plt.savefig("confusion_matrix_cnn.png")
plt.show()


# Definirea clasei dataset pentru testare
class AndraTestDfDataset(Dataset):
    def __init__(self, cale_fisierCSV, cale_imagini, img_preprocesare=None):
        self.tabelDATE = pd.read_csv(cale_fisierCSV)
        self.cale_imagini = cale_imagini
        self.img_preprocesare = img_preprocesare

    def __len__(self):
        return len(self.tabelDATE)

    def __getitem__(self, idx):
        img_id = self.tabelDATE.iloc[idx, 0]
        caleCompletaImagine = os.path.join(self.cale_imagini, f"{img_id}.png")
        inputImage = Image.open(caleCompletaImagine).convert("RGB")
        if self.img_preprocesare:# se aplica preprocesarea imaginilor definita mai sus
            inputImage = self.img_preprocesare(inputImage)
        return inputImage, img_id


# Incarcarea datelor de test
test_dataset = AndraTestDfDataset(
    cale_fisierCSV="test.csv",
    cale_imagini="test",
    img_preprocesare=img_preprocesare
)
test_loader = DataLoader(test_dataset, batch_size=DIMENSIUNE_batch, shuffle=False)

model.eval()
rezultateFINALE = []

with torch.no_grad():
    for imgIncarcatorAntrenare, image_ids in test_loader:
        imgIncarcatorAntrenare = imgIncarcatorAntrenare.to(dispozitivAAM)
        iesiriModel = model(imgIncarcatorAntrenare)
        _, etichetePrezise = torch.max(iesiriModel, 1)
        etichetePrezise = etichetePrezise.cpu().numpy()
        for img_id, label in zip(image_ids, etichetePrezise):
            rezultateFINALE.append((img_id, label))

# Scrierea rezultatelor intr-un fisier csv pentru submit
fisierFinalSubmitAAM = pd.DataFrame(
            rezultateFINALE,
            columns=["image_id", "label"]
            )
fisierFinalSubmitAAM.to_csv("submissionCNNfinal.csv", index=False) # salvarea fisierului
print("Fisierul 'submissionCNNfinal.csv' a fost creat!")