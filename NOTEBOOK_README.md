# Tutorial Completo Jupyter Notebook

## 📓 Introduzione

Il file `tutorial_completo.ipynb` è un notebook Jupyter completo e interattivo che integra tutte le funzionalità del progetto di monitoraggio della glicemia tramite ECG in un unico documento.

## 🎯 Scopo

Questo notebook è stato creato per permetterti di:
- **Imparare** tutti i concetti del progetto passo-passo
- **Eseguire** il codice interattivamente senza creare file separati
- **Visualizzare** risultati e grafici direttamente nel notebook
- **Sperimentare** con i parametri e vedere gli effetti in tempo reale

## 📋 Contenuto del Notebook

### Sezioni Principali

1. **Setup e Installazione**
   - Installazione dipendenze
   - Import librerie

2. **Caricamento Dati**
   - Generazione dati sintetici
   - Caricamento dataset D1NAMO reale

3. **Visualizzazione Dati**
   - Plot segnali ECG
   - Visualizzazione glucosio nel tempo

4. **Elaborazione Segnale ECG**
   - Preprocessing (filtraggio, normalizzazione)
   - Rilevamento punti fiduciali (P, Q, R, S, T)
   - Visualizzazione battiti con annotazioni

5. **Estrazione Features**
   - 35 features morfologiche per battito
   - 18 features HRV (Heart Rate Variability)
   - Visualizzazione distribuzioni features

6. **Addestramento Modelli**
   - Split temporale train-test
   - Training modello MBeat (Random Forest)
   - Valutazione metriche (AUC, Sensitivity, Specificity, F1)

7. **Visualizzazioni Avanzate**
   - Matrice di confusione
   - Curva ROC
   - Importanza features

8. **Pipeline Completa**
   - Workflow end-to-end automatizzato
   - Rilevamento ipoglicemia
   - Rilevamento iperglicemia

9. **Utilizzo Avanzato**
   - Confronto tra modelli
   - Salvataggio/caricamento modelli
   - Predizioni in tempo reale

10. **Best Practices**
    - Consigli pratici
    - Troubleshooting
    - Estensioni possibili

## 🚀 Come Utilizzare il Notebook

### Prerequisiti

```bash
# Installa Jupyter (se non già installato)
pip install jupyter ipykernel

# Installa le dipendenze del progetto
pip install -r requirements.txt
```

### Apertura del Notebook

**Opzione 1: Jupyter Notebook (classico)**
```bash
jupyter notebook tutorial_completo.ipynb
```

**Opzione 2: JupyterLab (moderno)**
```bash
jupyter lab tutorial_completo.ipynb
```

**Opzione 3: VS Code**
1. Installa l'estensione "Jupyter" in VS Code
2. Apri il file `tutorial_completo.ipynb`
3. Esegui le celle con Shift+Enter

**Opzione 4: Google Colab**
1. Carica il file su Google Drive
2. Apri con Google Colab
3. Nota: potrebbe essere necessario installare alcune dipendenze

### Esecuzione

1. **Esegui le celle in sequenza** dall'alto verso il basso (Shift+Enter)
2. **Non serve il dataset reale** - il notebook usa dati sintetici di default
3. **Ogni sezione è indipendente** - puoi saltare sezioni se già familiare

## 📊 Cosa Puoi Fare

### ✅ Funziona Out-of-the-Box

Il notebook è progettato per funzionare immediatamente senza configurazioni:
- ✅ Genera dati sintetici automaticamente
- ✅ Esegue tutto il workflow completo
- ✅ Crea visualizzazioni interattive
- ✅ Addestra e valuta modelli

### 🔧 Personalizzazioni Possibili

```python
# Modifica durata dati sintetici
patient_data = generator.generate_patient_data(duration_hours=4)  # invece di 2

# Cambia soglia ipoglicemia
threshold_hypo = 60  # invece di 70

# Modifica parametri modello
mbeat_model = MBeat(n_estimators=200, max_depth=10)  # invece dei default

# Usa dati reali
loader = D1NAMODataLoader(data_dir='data/raw')
patient_data = loader.load_patient_data('001')
```

## 📈 Output Attesi

Eseguendo tutte le celle del notebook otterrai:

1. **Grafici ECG**: Visualizzazione segnale grezzo e preprocessato
2. **Plot Glucosio**: Andamento temporale con soglie cliniche
3. **Analisi Features**: Distribuzioni e importanza features
4. **Metriche Modello**: AUC, Sensitivity, Specificity, F1-Score
5. **Visualizzazioni**: Matrice confusione, curva ROC
6. **Modello Salvato**: File `.pkl` nella directory `models/`

## 🆚 Differenze con gli Script

| Caratteristica | Notebook (`tutorial_completo.ipynb`) | Script (`example.py`, `train_all_patients.py`) |
|----------------|-------------------------------------|-----------------------------------------------|
| **Interattività** | ✅ Alta - esegui celle singolarmente | ❌ Bassa - esegue tutto in una volta |
| **Visualizzazioni** | ✅ Inline nel documento | ❌ Finestre separate o file |
| **Documentazione** | ✅ Integrata con spiegazioni | ❌ Solo commenti nel codice |
| **Apprendimento** | ✅ Ideale per imparare | ❌ Per utenti esperti |
| **Sperimentazione** | ✅ Facile modificare e rilanciare | ❌ Richiede modifica file |
| **Produzione** | ❌ Non ideale | ✅ Meglio per automazione |

## 💡 Consigli d'Uso

### Per Principianti

1. **Leggi le spiegazioni**: Ogni sezione ha descrizioni dettagliate in italiano
2. **Esegui in ordine**: Le celle dipendono dalle precedenti
3. **Sperimenta**: Modifica valori e riesegui per vedere gli effetti
4. **Non preoccuparti degli errori**: Puoi sempre ricominciare

### Per Utenti Avanzati

1. **Salta alle sezioni che ti interessano**
2. **Modifica il codice per i tuoi esperimenti**
3. **Usa come template per analisi personalizzate**
4. **Combina con dataset reali**

## 🔧 Troubleshooting

### Problema: Celle non eseguibili

**Causa**: Jupyter non installato o kernel non configurato

**Soluzione**:
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=signalproc
```

### Problema: Import errors

**Causa**: Dipendenze mancanti

**Soluzione**:
```bash
pip install -r requirements.txt
```

### Problema: Grafici non visualizzati

**Causa**: Backend matplotlib non configurato

**Soluzione**: Aggiungi all'inizio del notebook:
```python
%matplotlib inline
```

### Problema: Out of memory

**Causa**: Durata dati troppo lunga

**Soluzione**: Riduci `duration_hours` nella generazione dati:
```python
patient_data = generator.generate_patient_data(duration_hours=1)
```

## 📚 Risorse Aggiuntive

- **README.md**: Panoramica generale del progetto
- **USAGE_GUIDE.md**: Guida dettagliata all'utilizzo con dataset reale
- **example.py**: Script di esempio non interattivo
- **train_all_patients.py**: Training su tutti i pazienti del dataset

## 🎓 Obiettivi di Apprendimento

Completando questo notebook imparerai:

- ✅ Come processare segnali ECG biomedici
- ✅ Tecniche di estrazione features da segnali temporali
- ✅ Addestramento modelli ML per classificazione binaria
- ✅ Valutazione metriche clinicamente rilevanti
- ✅ Best practices per machine learning su dati biomedici
- ✅ Pipeline complete end-to-end per problemi reali

## 🤝 Contributi

Se trovi errori o hai suggerimenti per migliorare il notebook:
1. Apri una issue su GitHub
2. Proponi modifiche tramite pull request
3. Condividi feedback e suggerimenti

## 📄 Licenza

Questo notebook fa parte del progetto ECG-Based Glucose Monitoring ed è distribuito per scopi educativi e di ricerca.

---

**Buon lavoro con il tutorial! 🚀**

Se hai domande o problemi, consulta le sezioni di troubleshooting o apri una issue sul repository.
