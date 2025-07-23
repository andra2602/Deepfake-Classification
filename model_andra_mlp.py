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
# Sunt augumentate si normalizate [-1,1]
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
    

# Arhitectura MLP - laborator 6 + imbunatatiri
class AndraModelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial, imaginea (data noastra de intrare) este aplatizata intr-un vector 1-dimensional
        self.input_flattener = nn.Flatten()

        # Definirea straturilor ascunse cu o structura progesiva 1024 -> 512 -> 256 -> 5 pentru invatare treptata
        self.primul_hiddenLayer = nn.Linear(3 * 128 * 128, 1024) # trecem de la 3*128*128 pixeli (configuratia initiala) la 1024 neuroni
        self.alDoilea_hiddenLayer = nn.Linear(1024, 512)  # 1024 --> 512
        self.alTreilea_hiddenLayer = nn.Linear(512, 256)   # 512--> 256
        
        # Definesc acest dropout pentru a preveni sansa de overfitting
        # Astfel 30% din neuroni sunt dezactivati in mod aleator
        self.dropout = nn.Dropout(p=0.3) # am ales 0.3 deoarece l-am considerat a fi un dropout echilibrat 

        # Se defineste startul final care de la ultimii 256 de neuroni, proiecteaza catre cele 5 clase de predictie
        self.final_predictor = nn.Linear(256, NUMAR_clase)

        # Se defineste functia de activare LeakyReLU, cu un alpha mic pentru a evita neuroni morti
        # Am inlocuit ReLU cu LeakyReLU pentru ca aceasta permite un mic leak de informatie pentru valorile negative si previne problema neuronilor morti
        self.functieActivare = nn.LeakyReLU(0.01) 

    def forward(self, MLP_AAM_X):
        MLP_AAM_X = self.input_flattener(MLP_AAM_X) 
        # se splica cele 3 straturi , fiecare fiind activat de functia aleasa
        MLP_AAM_X = self.functieActivare(self.primul_hiddenLayer(MLP_AAM_X))
        MLP_AAM_X = self.functieActivare(self.alDoilea_hiddenLayer(MLP_AAM_X))
        MLP_AAM_X = self.functieActivare(self.alTreilea_hiddenLayer(MLP_AAM_X))
        # Se efectueaza un dropout pentru regularizare in timpul antrenarii
        MLP_AAM_X = self.dropout(MLP_AAM_X)
        MLP_AAM_X = self.final_predictor(MLP_AAM_X) # stratul final pentru clasificare, fara activare
        return MLP_AAM_X

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
model = AndraModelMLP().to(dispozitivAAM)
# Optimizatorul folosit este SGD, cu un learning rate fix si momentum
# Momentum accelereaza antrenarea si ajuta la evitarea stagnarii în minime locale
AAMoptimizator = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(AAMoptimizator, step_size=5, gamma=0.5) # Scheduler-ul scade learning rate-ul la fiecare 5 epoci, cu un factor de 0.5, il folosesc pentru a stabiliza invatarea

NUMAR_epoci = 15
for epocaCurenta in range(NUMAR_epoci):
    model.train() # se seteaza modelul in modul de antrenare
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
# retinem numarul de predictii corecte, numarul total de exmple evaluate si suma pierderilor
correct = 0 
total = 0
total_loss = 0

with torch.no_grad(): # In timpul evaluarii, se dezactiveaza calculul de gradient pentru a economisi resurse
    for imgIncarcatorAntrenare, labelsIncarcatorAntrenare in loaderPentruValidare:
        imgIncarcatorAntrenare, labelsIncarcatorAntrenare = imgIncarcatorAntrenare.to(dispozitivAAM), labelsIncarcatorAntrenare.to(dispozitivAAM)
        iesiriModel = model(imgIncarcatorAntrenare)
        # calcularea pierderii
        loss = funcPierdere(iesiriModel, labelsIncarcatorAntrenare)
        total_loss += loss.item()
        
        _, etichetePrezise = torch.max(iesiriModel, 1) 
        correct += (etichetePrezise == labelsIncarcatorAntrenare).sum().item() # vedem cate predictii au fost corecte
        total += labelsIncarcatorAntrenare.size(0)

# Se calculeaza si afiseaza acuratetea si pierderea medie
acurateteValidare = 100 * correct / total
lossMediu = total_loss / len(loaderPentruValidare)

print(f"\nPerformanta modelului MLP pe setul de validare: {acurateteValidare:.2f}%")
print(f"Loss-ul mediu la validare: {lossMediu:.4f}")

# Matricea de confuzie pentru modelul MLP

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
matriceConfuzieMLP = confusion_matrix(labelsMatriceConfuzie, preziceriMatriceConfuzie)
disp = ConfusionMatrixDisplay(confusion_matrix=matriceConfuzieMLP, display_labels=[0, 1, 2, 3, 4])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de Confuzie - MLP")
plt.savefig("confusion_matrix_mlp_final.png")
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
        inputImagine = Image.open(caleCompletaImagine).convert("RGB")
        if self.img_preprocesare: # se aplica preprocesarea imaginilor definita mai sus
            inputImagine = self.img_preprocesare(inputImagine)
        return inputImagine, img_id


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
fisierFinalSubmitAAM.to_csv("submissionMLPfinal.csv", index=False) # salvarea fisierului
print("Fisierul 'submissionMLPfinal.csv' a fost creat!")