# Named Entity Recognition (NER) Project A complete NLP pipeline that trains and evaluates a custom Named Entity Recognition model using the CoNLL-2003 English dataset. The project includes both rule-based and model-based approaches, evaluation metrics, and visualizations to analyze entity extraction performance.

## Project Overview This project implements an end-to-end NER system using the CoNLL-2003 English dataset from Kaggle. It compares a rule-based method with a trained spaCy model and provides evaluation, visualization, and persistence of results.

### Dataset - **Source**: [CoNLL-2003 English Dataset (Kaggle)](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)

* **Format**: CoNLL-style IOB tagging
* **Training samples**: \~14,000 sentences
* **Validation samples**: \~3,500 sentences
* **Test samples**: \~3,500 sentences
* **Entity categories**: 4 classes

  * PER (Person)
  * ORG (Organization)
  * LOC (Location)
  * MISC (Miscellaneous)

## Features Implemented  

### Core Requirements  
- ✅ **Data Processing**: Parsing CoNLL IOB files and converting them to spaCy training format  
- ✅ **Rule-Based NER**: Using spaCy’s `EntityRuler` to learn from training patterns  
- ✅ **Model-Based NER**: Training a custom spaCy model with a validation set  
- ✅ **Evaluation**: Precision, Recall, F1-score, and per-entity metrics  
- ✅ **Visualization**: Entity highlighting via displaCy, confusion matrix plots 

### Bonus Features  
- ✅ **Model Comparison**: Compare spaCy baseline models (`en_core_web_sm` vs `en_core_web_md`)  
- ✅ **Persistence**: Save models, metrics, confusion matrix, and HTML visualizations  
- ✅ **Comprehensive Output**: JSON reports, visualizations, and summaries  

## Technology Stack ### Libraries Used 
- **NLP**: spaCy 
- **Evaluation**: scikit-learn 
- **Visualization**: matplotlib, seaborn, spaCy displaCy 
- **Utilities**: json, tqdm, os

### Models Implemented 
1. **Rule-Based NER** - EntityRuler patterns built from training set 
2. **Custom spaCy Model** - Trained on CoNLL-2003 dataset with dropout and iterations 
3. **Baseline spaCy Models** - Pre-trained `en_core_web_sm` / `en_core_web_md`

## Project Structure

named-entity-recognition/
│
├── Dataset/
│   ├── train.txt              # Training data
│   ├── valid.txt              # Validation data
│   └── test.txt               # Test data
│
├── ner\_results/
│   ├── trained\_ner\_model/     # Saved spaCy model
│   ├── evaluation\_results.json
│   ├── confusion\_matrix.png
│   ├── model\_comparison.png
│   └── ner\_visualization\_\*.html
│
├── ner\_pipeline.py            # Main pipeline script
├── requirements.txt           # Dependencies
└── README.md                  # This file

## Installation & Setup 
### Prerequisites - Python 3.8+ - 4GB+ RAM (recommended for training)

### Installation Steps 1. **Clone the repository**

```bash
git clone <repository-url>
cd named-entity-recognition
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download spaCy model**

```bash
python -m spacy download en_core_web_sm
```

## Usage 
### Training and Evaluation Run the pipeline:

```bash
python ner_pipeline.py
```

### Making Predictions Load your trained model:

```python
import spacy
nlp = spacy.load("ner_results/trained_ner_model")
doc = nlp("Apple is planning to open a new office in Paris.")
print([(ent.text, ent.label_) for ent in doc.ents])
```

## Model Performance  

| Model                      | Precision | Recall | F1-Score |
|-----------------------------|-----------|--------|----------|
| **Rule-Based NER**          | ~0.39     | ~0.64  | ~0.48    |
| **Custom spaCy Model**      | ~0.82     | ~0.80  | ~0.81    |
| **Pretrained (en_core_web_sm)** | ~0.78 | ~0.75  | ~0.76    |
| **Pretrained (en_core_web_md)** | ~0.80 | ~0.78  | ~0.79    |

**✨ Best Model:** Custom spaCy Model with **F1 ≈ 0.81**

## Key Technical Decisions 
### Data Handling 
- Converted CoNLL IOB format into spaCy’s `Doc` and training data 
- Implemented robust parsing for edge cases and invalid spans ### Training Choices 
- Used dropout for regularization 
- Iterative updates with validation for early stopping 
- Balanced entity type evaluation ### Evaluation 
- Used `sklearn.metrics` for precision, recall, F1 - Added confusion matrix and entity
-level scores

## Insights & Analysis 
- **Strongest entity type**: LOC (locations, F1 \~0.85) 
- **Hardest entity type**: MISC (ambiguous, F1 \~0.74) 
- **Rule-based NER**: Captures frequent entities but misses context 
- **Custom model**: Achieves highest overall balance of precision and recall

## Future Improvements 
### Potential Enhancements 
- **Transformer-based models**: Integrate `en_core_web_trf` (BERT) 
- **Hyperparameter tuning**: Batch size, learning rate optimization 
- **Multilingual NER**: Expand beyond English 
- **Deployment**: REST API for real-time entity extraction ### Deployment Considerations 
- **Containerization**: Docker packaging 
- **Monitoring**: Track F1 drift in production 
- **Interactive UI**: Streamlit/Gradio for demos

## Contributing 
1. Fork the repository 
2. Create feature branch (`git checkout -b feature/improvement`) 
3. Commit changes (`git commit -am 'Add improvement'`) 
4. Push to branch (`git push origin feature/improvement`) 
5. Create Pull Request

## License This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments 
- **Dataset**: CoNLL-2003 (English version) from Kaggle 
- **Libraries**: spaCy, scikit-learn, matplotlib, seaborn 
- **Research Inspiration**: NER best practices in NLP

## Contact For questions or suggestions, please open an issue on GitHub.

\--- *Project completed as part of NLP Named Entity Recognition study.*
