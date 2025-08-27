# Named Entity Recognition (NER) Project

## 📌 Overview
This project implements a **Named Entity Recognition (NER)** system using **spaCy**.  
It is designed to identify and classify named entities such as **persons, organizations, locations, dates, etc.** in text.

## 🚀 Features
- Train a custom NER model using spaCy.
- Visualize entity predictions with `displacy`.
- Evaluate model performance on test examples.
- Support for fine-tuning with custom datasets.

## 📂 Project Structure
```
NamedEntityRecognition/
│── data/                # Training and test datasets
│── ner_trained_model/   # Saved spaCy trained model
│── scripts/             # Python scripts for training & evaluation
│── README.txt           # Project documentation
```
- **train.py** → Script to train the NER model.  
- **evaluate.py** → Script to test and evaluate the trained model.  
- **visualize.py** → Script to visualize named entities with `displacy`.

## 🛠️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/mariemwalid19/NamedEntityRecognition.git
   cd NamedEntityRecognition
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate    # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
### Train the Model
```bash
python scripts/train.py
```

### Evaluate on Test Data
```bash
python scripts/evaluate.py
```

### Visualize Named Entities
```bash
python scripts/visualize.py
```

## 📊 Example Output
**Input Text:**  
`"Barack Obama was born in Hawaii and served as the 44th President of the United States."`

**Predicted Entities:**  
- `Barack Obama` → PERSON  
- `Hawaii` → LOCATION  
- `44th President` → TITLE  
- `United States` → LOCATION

## 🔮 Future Improvements
- Add support for multi-language NER.
- Improve accuracy using transformer-based models (BERT, RoBERTa, etc.).
- Build a web interface for live NER predictions.

## 👩‍💻 Author
- **Mariem Walid**  
Fourth-year Computer Science student, Software Engineer, and AI Engineer.  
Passionate about NLP, Data Science, and Machine Learning.

---
✨ *This project was built for learning and experimentation with spaCy NER.*
