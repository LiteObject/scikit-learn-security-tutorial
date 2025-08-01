# Scikit-Learn Tutorial: Building Network Security AI Models

**Learn Machine Learning by Building Real Network Security Models**

This tutorial teaches you scikit-learn fundamentals by creating the same AI models used in [NetGuardian-AI](https://github.com/LiteObject/NetGuardian-AI). You'll build three different types of machine learning models from scratch and understand how they work together for network security analysis.

## 📁 **Project Structure**

```
scikit-learn-security-tutorial/
├── 📄 README.md                   # This tutorial guide
├── 🐍 data_generator.py           # Network data generation and feature engineering
├── 🤖 model_trainer.py            # Machine learning model training
├── 📊 model_evaluator.py          # Advanced analysis and visualizations
├── 🎯 network_security_ml.py      # Main application with CLI
├── 🧪 test_components.py          # Component testing script
├── ⚙️ setup.py                    # Automated setup and verification
├── 📦 requirements.txt            # Python dependencies
├── 🪟 setup_windows.bat           # Windows setup script
└── 🐧 setup_unix.sh               # Unix/Linux/macOS setup script
```

## 🎯 **What You'll Learn**

- ✅ **Supervised Learning**: Classification and Regression
- ✅ **Unsupervised Learning**: Anomaly Detection
- ✅ **Feature Engineering**: Converting raw data to ML features
- ✅ **Model Training**: Fitting models to data
- ✅ **Model Evaluation**: Testing and validation
- ✅ **Real-World Application**: Network security use case

## 📋 **Prerequisites**

- Basic Python knowledge (variables, functions, loops)
- Understanding of lists and dictionaries
- No prior machine learning experience needed!

## 🚀 **Quick Start**

### **Option 1: Automated Setup (Recommended)**

#### **Windows Users:**
```cmd
# Clone and setup
git clone https://github.com/LiteObject/scikit-learn-security-tutorial.git
cd scikit-learn-security-tutorial
setup_windows.bat
```

#### **macOS/Linux Users:**
```bash
# Clone and setup
git clone https://github.com/LiteObject/scikit-learn-security-tutorial.git
cd scikit-learn-security-tutorial
chmod +x setup_unix.sh
./setup_unix.sh
```

### **Option 2: Manual Setup**

```bash
# Clone the repository
git clone https://github.com/LiteObject/scikit-learn-security-tutorial.git
cd scikit-learn-security-tutorial

# Create virtual environment
python -m venv tutorial_env

# Activate environment
# Windows:
tutorial_env\Scripts\activate
# macOS/Linux:
source tutorial_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup.py
```

## 🎮 **Running the Tutorial**

### **Basic Tutorial**
```bash
python network_security_ml.py
```

### **Interactive Mode (Step-by-step learning)**
```bash
python network_security_ml.py --mode interactive
```

### **Advanced with Visualizations**
```bash
python network_security_ml.py --mode advanced --visualize
```

### **Test Individual Components**
```bash
python test_components.py
```

## 🧪 **What Each File Does**

| File | Purpose | Key Features |
|------|---------|--------------|
| `data_generator.py` | **Data Creation** | Generates realistic network device data, feature engineering |
| `model_trainer.py` | **ML Training** | 3 models: Classification, Regression, Anomaly Detection |
| `model_evaluator.py` | **Analysis** | Confusion matrices, feature importance, learning curves |
| `network_security_ml.py` | **Main App** | CLI interface, interactive mode, custom testing |
| `test_components.py` | **Testing** | Verify all components work correctly |

## � **Tutorial Learning Path**

### **🎯 Step 1: Understanding the Data** (`data_generator.py`)
Learn how to convert raw network data into machine learning features:
- **Feature Engineering**: Transform port numbers into numerical features
- **Data Generation**: Create realistic synthetic training data
- **Device Patterns**: Model different types of network devices

**Key Concept:** Feature engineering is the most important part of machine learning!

### **🤖 Step 2: Building ML Models** (`model_trainer.py`)
Build three different types of machine learning models:

1. **🎯 Device Classification** (Random Forest)
   - **Purpose**: Identify device types (IoT, Server, PC, etc.)
   - **Algorithm**: Random Forest Classifier
   - **Output**: Device category with confidence score

2. **🕵️ Anomaly Detection** (Isolation Forest)  
   - **Purpose**: Find unusual/suspicious network behavior
   - **Algorithm**: Isolation Forest
   - **Output**: Normal vs Anomalous classification

3. **⚡ Risk Assessment** (Random Forest Regression)
   - **Purpose**: Predict security risk score (0.0 to 1.0)
   - **Algorithm**: Random Forest Regressor
   - **Output**: Continuous risk score

### **📊 Step 3: Model Analysis** (`model_evaluator.py`)
Understand how well your models perform:
- **Confusion Matrix**: Which device types get confused?
- **Feature Importance**: Which network patterns matter most?
- **Learning Curves**: Do you need more training data?
- **Risk Accuracy**: How accurate are risk predictions?

### **🎮 Step 4: Interactive Testing** (`network_security_ml.py`)
Test your models on real scenarios:
- **Pre-built Examples**: Web servers, suspicious devices, IoT devices
- **Custom Testing**: Enter your own port combinations
- **Real-time Analysis**: See all three models work together

## 💡 **Key Learning Concepts**

| Concept | What You'll Learn | Real-World Application |
|---------|------------------|----------------------|
| **Feature Engineering** | Convert raw data to ML features | Essential for any ML project |
| **Classification** | Predict categories | Device identification, spam detection |
| **Regression** | Predict continuous values | Risk scores, price prediction |
| **Anomaly Detection** | Find unusual patterns | Fraud detection, security monitoring |
| **Model Evaluation** | Measure performance | Ensuring models work in production |

## 🔬 **Sample Output**

When you run the tutorial, you'll see analysis like this:

```
🤖 AI Analysis Results:
   Device Type: Linux Server (confidence: 0.892)
   Risk Score: 0.423 (MEDIUM)
   Anomaly Status: NORMAL (score: 0.156)
```

## 🛠️ **Example Code Snippets**

The tutorial includes complete working examples. Here are some highlights:
            float(has_rdp),      # [5] Has RDP (risky!)
            float(has_smb),      # [6] Has SMB (risky!)
            float(has_ftp),      # [7] Has FTP (risky!)
            float(port_spread),  # [8] Port range
            float(high_ports)    # [9] High port count
        ]
    
    def generate_sample(self, device_type: int) -> Tuple[List[float], int, float]:
        """Generate one realistic training sample"""
        pattern = self.device_patterns[device_type]
        
        # Start with typical ports for this device type
        ports = pattern["typical_ports"].copy()
        
        # Add realistic noise (30% chance of additional ports)
        if random.random() > 0.7:
            # Add 1-3 random ports to simulate real-world variation
            extra_ports = random.sample(range(1024, 65535), random.randint(1, 3))
            ports.extend(extra_ports)
        
        # Extract features
        features = self.extract_features(ports)
        
        # Generate risk score with some noise
        base_risk = pattern["base_risk"]
        # Add Gaussian noise (normal distribution) to make it realistic
        risk_noise = random.gauss(0, 0.1)  # mean=0, std=0.1
        risk_score = max(0.0, min(1.0, base_risk + risk_noise))
        
        return features, device_type, risk_score
    
    def generate_dataset(self, samples_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a complete training dataset"""
        
        all_features = []
        all_device_labels = []
        all_risk_scores = []
        
        print("🏭 Generating training data...")
        print(f"📊 Creating {samples_per_class} samples per device type...")
        
        # Generate samples for each device type
        for device_type in range(6):  # 0-5 device types
            device_name = self.device_patterns[device_type]["name"]
            print(f"   📱 Generating {device_name} samples...")
            
            for _ in range(samples_per_class):
                features, label, risk = self.generate_sample(device_type)
                all_features.append(features)
                all_device_labels.append(label)
                all_risk_scores.append(risk)
        
        # Convert to numpy arrays (required by scikit-learn)
        X = np.array(all_features)      # Features: shape (600, 10)
        y_device = np.array(all_device_labels)  # Device labels: shape (600,)
        y_risk = np.array(all_risk_scores)      # Risk scores: shape (600,)
        
        print(f"✅ Generated {len(X)} total samples")
        print(f"📏 Feature matrix shape: {X.shape}")
        print(f"🏷️ Device labels shape: {y_device.shape}")
        print(f"⚡ Risk scores shape: {y_risk.shape}")
        
        return X, y_device, y_risk
    
    def get_device_name(self, device_type: int) -> str:
        """Get human-readable device name"""
        return self.device_patterns[device_type]["name"]

# Test the data generator
if __name__ == "__main__":
    generator = NetworkDataGenerator()
### **Feature Engineering Example:**
```python
# Convert network ports to ML features
ports = [22, 80, 443, 3389]  # SSH, HTTP, HTTPS, RDP
features = generator.extract_features(ports)
# Output: [4.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3367.0, 0.0]
```

### **Model Training Example:**
```python
# Train device classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train_scaled, y_device_train)

# Test accuracy
predictions = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_device_test, predictions)
print(f"Accuracy: {accuracy:.3f}")
```

### **Prediction Example:**
```python
# Analyze new device
new_ports = [23, 21, 3389, 445]  # Suspicious combination
features = extract_features(new_ports)
device_type = classifier.predict([features])[0]
risk_score = risk_predictor.predict([features])[0]
print(f"Device: {get_device_name(device_type)}, Risk: {risk_score:.3f}")
```

## 🎓 **Learning Outcomes**

After completing this tutorial, you'll understand:

### **🧠 Machine Learning Concepts**
- **Supervised vs Unsupervised Learning**
- **Classification vs Regression** 
- **Feature Engineering** - Converting raw data to ML features
- **Training vs Testing** - Avoiding overfitting
- **Model Evaluation** - Accuracy, precision, recall

### **🛠️ Scikit-Learn Skills**
- **RandomForestClassifier** - Multi-class classification
- **RandomForestRegressor** - Continuous value prediction  
- **IsolationForest** - Anomaly detection
- **StandardScaler** - Feature normalization
- **train_test_split** - Data splitting
- **Model evaluation metrics**

### **🌐 Real-World Application**
- **Network Security** use case
- **Feature extraction** from network data
- **Multi-model systems** working together
- **Practical AI implementation**

## 🔬 **Experiments to Try**

### **1. Different Algorithms**
```python
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Try Support Vector Machine
svm_classifier = SVC(kernel='rbf', random_state=42)

# Try Naive Bayes
nb_classifier = GaussianNB()

# Try Neural Network
nn_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
```

### **2. Enhanced Features**
```python
def enhanced_features(ports):
    """Add more sophisticated features"""
    basic_features = extract_features(ports)
    
    # Add new features
    system_ports = len([p for p in ports if p < 1024])
    web_services = 1 if any(p in [80, 8080, 443, 8443] for p in ports) else 0
    remote_access = 1 if any(p in [21, 22, 23, 3389] for p in ports) else 0
    
    return basic_features + [system_ports, web_services, remote_access]
```

### **3. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Test model with cross-validation
cv_scores = cross_val_score(classifier, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f}")
```

## 🏆 **Challenge Projects**

Now that you understand the basics, try building models for different problems:

1. **📧 Email Spam Detection** - Classify emails as spam/not spam
2. **📈 Stock Price Prediction** - Predict stock movements  
3. **🖼️ Image Classification** - Classify handwritten digits
4. **💬 Sentiment Analysis** - Analyze text sentiment
5. **🏥 Medical Diagnosis** - Predict disease from symptoms
6. **🛒 Recommendation System** - Suggest products to users

## 🛠️ **Troubleshooting**

### **Common Issues:**

**❌ Import Error: No module named 'sklearn'**
```bash
pip install scikit-learn
```

**❌ ModuleNotFoundError: No module named 'seaborn'**
```bash
pip install seaborn matplotlib
```

**❌ Virtual environment issues**
```bash
# Recreate environment
rm -rf tutorial_env  # or rmdir /s tutorial_env on Windows
python -m venv tutorial_env
source tutorial_env/bin/activate  # or tutorial_env\Scripts\activate on Windows
pip install -r requirements.txt
```

**❌ Models showing poor performance**
- Try increasing `samples_per_class` in data generation
- Check if data is properly scaled
- Verify feature engineering is working correctly

---

## 📚 **Further Learning Resources**

### **📖 Books**
- "Hands-On Machine Learning" by Aurélien Géron
- "Python Machine Learning" by Sebastian Raschka
- "Introduction to Statistical Learning" (free PDF)

### **🌐 Online Courses**
- Coursera: Machine Learning by Andrew Ng
- edX: MIT Introduction to Machine Learning
- Kaggle Learn: Free micro-courses

### **🛠️ Practice Datasets**
- Kaggle competitions
- UCI Machine Learning Repository
- scikit-learn built-in datasets

---

## 🎯 **Challenge: Build Your Own Model**

Now that you understand the basics, try building a model for a different problem:

1. **Email Spam Detection** - Classify emails as spam/not spam
2. **Stock Price Prediction** - Predict stock movements
3. **Image Classification** - Classify handwritten digits
4. **Sentiment Analysis** - Analyze text sentiment


---

**🎓 Congratulations! You've learned scikit-learn by building real AI models for network security. The concepts you've learned here apply to any machine learning problem. Keep experimenting and building!**
