# Scikit-Learn Network Security Tutorial

**Learn machine learning by building actual network security models**

This tutorial teaches scikit-learn basics by creating real AI models for network security analysis. You'll build three different machine learning models and see how they work together to identify devices and detect threats.

## What is scikit-learn?

Scikit-learn is basically a Python library that makes machine learning accessible. Think of it like having a Swiss Army knife for AI - it has tools for classification (sorting things into categories), regression (predicting numbers), anomaly detection (finding weird stuff), and data processing.

Here's how simple it is:

```python
from sklearn.ensemble import RandomForestClassifier

# Create a model
model = RandomForestClassifier()

# Train it with your data
model.fit(training_data, labels)

# Make predictions
prediction = model.predict(new_data)
```

That's it. No PhD required.

In this tutorial, you'll use scikit-learn to build models that can identify different types of network devices and detect security threats - the same kind of stuff cybersecurity professionals use.

## Project files

```
├── README.md                    # You are here
├── docs/                        
│   ├── supervised-vs-unsupervised.md    # ML concepts explained
│   ├── classification-vs-regression.md  # Types of predictions
│   └── feature-engineering.md           # Data prep basics
├── data_generator.py            # Creates fake network data for training
├── model_trainer.py             # Where the ML magic happens
├── model_evaluator.py           # Tests how good your models are
├── network_security_ml.py       # Main app you'll actually run
├── test_components.py           # Quick tests to make sure stuff works
├── requirements.txt             # Python packages you need
└── setup scripts               # Makes installation easier
```

## What you'll learn

- Classification and regression models
- Anomaly detection 
- How to turn raw data into features
- Training and evaluating models
- Real network security applications

**New to machine learning?** Check out the guides in the `docs/` folder first. They explain the basics without the jargon:
- [Supervised vs Unsupervised Learning](docs/supervised-vs-unsupervised.md) 
- [Classification vs Regression](docs/classification-vs-regression.md)
- [Feature Engineering](docs/feature-engineering.md)

## Prerequisites

- Basic Python (variables, functions, loops)
- Know what lists and dictionaries are
- That's it - no ML experience needed

## Getting started

**Easy way (recommended):**

Windows:
```cmd
git clone https://github.com/LiteObject/scikit-learn-security-tutorial.git
cd scikit-learn-security-tutorial
setup_windows.bat
```

Mac/Linux:
```bash
git clone https://github.com/LiteObject/scikit-learn-security-tutorial.git
cd scikit-learn-security-tutorial
chmod +x setup_unix.sh
./setup_unix.sh
```

**Manual way:**
```bash
python -m venv tutorial_env
source tutorial_env/bin/activate  # Windows: tutorial_env\Scripts\activate
pip install -r requirements.txt
python setup.py
```

## Running the tutorial

Basic version:
```bash
python network_security_ml.py
```

Step-by-step interactive version (better for learning):
```bash
python network_security_ml.py --mode interactive
```

Full version with charts and graphs:
```bash
python network_security_ml.py --mode advanced --visualize
```

Test individual parts:
```bash
python test_components.py
```

## How it works

**Step 1: Data preparation** (`data_generator.py`)
Takes network port data and converts it into numerical features. For example, a web server with ports [22, 80, 443] becomes a list of 10 numbers that describe its characteristics.

**Step 2: Model training** (`model_trainer.py`)
Trains three different models:
- Random Forest for device classification
- Isolation Forest for anomaly detection  
- Random Forest Regressor for risk scoring

**Step 3: Evaluation** (`model_evaluator.py`)
Tests the models and shows you confusion matrices, feature importance, and learning curves so you can see what's working and what isn't.

**Step 4: Real-world testing** (`network_security_ml.py`)
Lets you test the models on actual scenarios. You can input port combinations and see what the AI thinks about them.

## Example output

When you run it, you'll see something like:

```
Analyzing device with ports: [22, 80, 443, 3389]

AI Analysis Results:
   Device Type: Linux Server (89% confidence)
   Risk Score: 0.42 (MEDIUM risk)
   Anomaly Status: NORMAL
```

## Common issues

**"No module named sklearn"**
```bash
pip install scikit-learn
```

**Charts not showing**
```bash
pip install matplotlib seaborn
```

**Models performing poorly**
Try generating more training data by increasing `samples_per_class` in the data generator.

## What's next?

Once you get this working, try modifying it:
- Add new device types
- Create different features
- Try other algorithms (SVM, Neural Networks, etc.)
- Apply the same approach to different problems (spam detection, stock prediction, etc.)

The concepts you learn here work for pretty much any machine learning problem.

## Background reading

**New to machine learning?**
- [Supervised vs Unsupervised Learning](docs/supervised-vs-unsupervised.md) - explains the basic types
- [Classification vs Regression](docs/classification-vs-regression.md) - when to use each approach
- [Feature Engineering](docs/feature-engineering.md) - how to prepare your data

**Want to learn more about scikit-learn?**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - official documentation with tutorials and API reference

---

That's it. Have fun building AI models that actually do something useful.
