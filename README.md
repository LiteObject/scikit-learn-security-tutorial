# Scikit-Learn Tutorial: Building Network Security AI Models

**Learn Machine Learning by Building Real Network Security Models**

This tutorial teaches you scikit-learn fundamentals by creating the same AI models used in [NetGuardian-AI](https://github.com/LiteObject/NetGuardian-AI). You'll build three different types of machine learning models from scratch and understand how they work together for network security analysis.

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

## 🚀 **Setup Your Learning Environment**

### **Step 1: Create a New Project**

```bash
# Create a new directory for your tutorial
mkdir scikit-learn-tutorial
cd scikit-learn-tutorial

# Create a Python virtual environment
python -m venv tutorial_env

# Activate the environment
# Windows:
tutorial_env\Scripts\activate
# macOS/Linux:
source tutorial_env/bin/activate
```

### **Step 2: Install Dependencies**

```bash
# Install required packages
pip install scikit-learn numpy pandas matplotlib seaborn

# Verify installation
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"
```

### **Step 3: Create Project Structure**

```bash
# Create project files
touch network_security_ml.py
touch data_generator.py
touch model_trainer.py
touch model_evaluator.py
```

## 📊 **Part 1: Understanding the Data**

### **Step 1: Create the Data Generator** (`data_generator.py`)

```python
import random
import numpy as np
from typing import List, Dict, Tuple

class NetworkDataGenerator:
    """Generates realistic network security training data"""
    
    def __init__(self):
        # Define realistic device patterns based on real networks
        self.device_patterns = {
            0: {
                "name": "IoT Device", 
                "typical_ports": [80],
                "base_risk": 0.3,
                "description": "Smart home devices, cameras, sensors"
            },
            1: {
                "name": "Linux Server", 
                "typical_ports": [22, 80, 443],
                "base_risk": 0.4,
                "description": "Web servers, application servers"
            },
            2: {
                "name": "Windows PC", 
                "typical_ports": [135, 3389, 445],
                "base_risk": 0.6,
                "description": "Desktop computers, workstations"
            },
            3: {
                "name": "Printer", 
                "typical_ports": [631, 9100],
                "base_risk": 0.2,
                "description": "Network printers, scanners"
            },
            4: {
                "name": "Router/Gateway", 
                "typical_ports": [22, 80, 443, 8080],
                "base_risk": 0.5,
                "description": "Network infrastructure devices"
            },
            5: {
                "name": "Vulnerable Device", 
                "typical_ports": [23, 21, 3389, 445],
                "base_risk": 0.9,
                "description": "Devices with high-risk services"
            }
        }
    
    def extract_features(self, ports: List[int]) -> List[float]:
        """
        Convert raw port data into numerical features for machine learning
        
        This is called FEATURE ENGINEERING - the most important part of ML!
        """
        if not ports:
            return [0.0] * 10
        
        # Feature 0: Total number of open ports
        num_ports = len(ports)
        
        # Features 1-7: Presence of specific important ports (binary features)
        has_ssh = 1 if 22 in ports else 0        # SSH (secure remote access)
        has_http = 1 if 80 in ports else 0       # HTTP (web server)
        has_https = 1 if 443 in ports else 0     # HTTPS (secure web)
        has_telnet = 1 if 23 in ports else 0     # Telnet (insecure!)
        has_rdp = 1 if 3389 in ports else 0      # Remote Desktop
        has_smb = 1 if 445 in ports else 0       # File sharing
        has_ftp = 1 if 21 in ports else 0        # File transfer
        
        # Feature 8: Port range spread (max - min)
        port_spread = max(ports) - min(ports) if len(ports) > 1 else 0
        
        # Feature 9: Number of high ports (> 1024)
        high_ports = len([p for p in ports if p > 1024])
        
        # Return as list of floats (required by scikit-learn)
        return [
            float(num_ports),    # [0] Total ports
            float(has_ssh),      # [1] Has SSH
            float(has_http),     # [2] Has HTTP
            float(has_https),    # [3] Has HTTPS
            float(has_telnet),   # [4] Has Telnet (risky!)
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
    
    # Test feature extraction
    test_ports = [22, 80, 443, 3389]  # SSH + HTTP + HTTPS + RDP
    features = generator.extract_features(test_ports)
    print(f"Test ports {test_ports} -> Features: {features}")
    
    # Generate small dataset
    X, y_device, y_risk = generator.generate_dataset(samples_per_class=10)
    print(f"\nSample features (first 3 rows):")
    print(X[:3])
    print(f"\nSample device labels (first 10): {y_device[:10]}")
    print(f"Sample risk scores (first 10): {y_risk[:10]}")
```

**🎓 Key Learning Points:**
- **Feature Engineering**: Converting raw data (port numbers) into numerical features
- **Data Generation**: Creating realistic synthetic data for training
- **NumPy Arrays**: Scikit-learn requires data as NumPy arrays

---

## 🤖 **Part 2: Building Your First Model - Device Classification**

### **Step 2: Create the Model Trainer** (`model_trainer.py`)

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import NetworkDataGenerator

class NetworkSecurityMLTutorial:
    """Learn scikit-learn by building network security models"""
    
    def __init__(self):
        self.data_generator = NetworkDataGenerator()
        
        # Initialize models (we'll train these step by step)
        self.device_classifier = None
        self.anomaly_detector = None
        self.risk_predictor = None
        self.feature_scaler = StandardScaler()
        
        # Store data for analysis
        self.X_train = None
        self.X_test = None
        self.y_device_train = None
        self.y_device_test = None
        self.y_risk_train = None
        self.y_risk_test = None
    
    def step_1_generate_and_explore_data(self):
        """Step 1: Generate data and explore it"""
        print("🎯 STEP 1: DATA GENERATION AND EXPLORATION")
        print("=" * 50)
        
        # Generate dataset
        X, y_device, y_risk = self.data_generator.generate_dataset(samples_per_class=100)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total samples: {len(X)}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Device types: {len(np.unique(y_device))}")
        
        # Show feature names for understanding
        feature_names = [
            "Total Ports", "Has SSH", "Has HTTP", "Has HTTPS", 
            "Has Telnet", "Has RDP", "Has SMB", "Has FTP",
            "Port Spread", "High Ports"
        ]
        
        print(f"\n🔧 Features we're using:")
        for i, name in enumerate(feature_names):
            print(f"   [{i}] {name}")
        
        # Show some actual data samples
        print(f"\n📝 Sample Data (first 3 devices):")
        for i in range(3):
            device_name = self.data_generator.get_device_name(y_device[i])
            print(f"   Sample {i+1}: {device_name}")
            print(f"      Features: {X[i]}")
            print(f"      Risk Score: {y_risk[i]:.3f}")
        
        # Split data for training and testing
        self.X_train, self.X_test, self.y_device_train, self.y_device_test, \
        self.y_risk_train, self.y_risk_test = train_test_split(
            X, y_device, y_risk, 
            test_size=0.2,      # 20% for testing
            random_state=42,    # Reproducible results
            stratify=y_device   # Ensure balanced split across device types
        )
        
        print(f"\n✂️ Data Split:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Testing samples: {len(self.X_test)}")
        
        return X, y_device, y_risk
    
    def step_2_build_device_classifier(self):
        """Step 2: Build and train device classification model"""
        print("\n🎯 STEP 2: DEVICE CLASSIFICATION MODEL")
        print("=" * 50)
        
        print("🌳 Building Random Forest Classifier...")
        print("   Random Forest = Many decision trees voting together")
        
        # Create the model with specific parameters
        self.device_classifier = RandomForestClassifier(
            n_estimators=100,    # Number of trees in the forest
            random_state=42,     # For reproducible results
            max_depth=10,        # Maximum depth of each tree
            min_samples_split=5, # Minimum samples to split a node
            min_samples_leaf=2   # Minimum samples in a leaf
        )
        
        print(f"   🌲 Forest size: {self.device_classifier.n_estimators} trees")
        print(f"   📏 Max depth: {self.device_classifier.max_depth}")
        
        # Scale features (normalize them)
        print("📊 Scaling features...")
        X_train_scaled = self.feature_scaler.fit_transform(self.X_train)
        X_test_scaled = self.feature_scaler.transform(self.X_test)
        
        # Train the model
        print("🎓 Training the model...")
        self.device_classifier.fit(X_train_scaled, self.y_device_train)
        print("✅ Model training complete!")
        
        # Test the model
        print("\n🧪 Testing the model...")
        predictions = self.device_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(self.y_device_test, predictions)
        
        print(f"🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Show detailed results
        device_names = [self.data_generator.get_device_name(i) for i in range(6)]
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(self.y_device_test, predictions, 
                                  target_names=device_names))
        
        # Show feature importance
        feature_names = [
            "Total Ports", "Has SSH", "Has HTTP", "Has HTTPS", 
            "Has Telnet", "Has RDP", "Has SMB", "Has FTP",
            "Port Spread", "High Ports"
        ]
        
        importances = self.device_classifier.feature_importances_
        print(f"\n🔥 Feature Importance (what the model cares about most):")
        for name, importance in zip(feature_names, importances):
            print(f"   {name}: {importance:.3f}")
        
        return accuracy
    
    def step_3_build_anomaly_detector(self):
        """Step 3: Build anomaly detection model"""
        print("\n🎯 STEP 3: ANOMALY DETECTION MODEL")
        print("=" * 50)
        
        print("🕵️ Building Isolation Forest for anomaly detection...")
        print("   Isolation Forest = Finds data points that are 'easy to isolate'")
        print("   Anomalies = Things that don't fit normal patterns")
        
        # Create the anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,   # Expect 10% of data to be anomalous
            random_state=42,
            n_estimators=100     # Number of isolation trees
        )
        
        print(f"   🌲 Number of isolation trees: {self.anomaly_detector.n_estimators}")
        print(f"   ⚠️ Expected contamination: {self.anomaly_detector.contamination * 100}%")
        
        # Train on scaled training data
        X_train_scaled = self.feature_scaler.transform(self.X_train)
        
        print("🎓 Training anomaly detector...")
        self.anomaly_detector.fit(X_train_scaled)
        print("✅ Anomaly detector training complete!")
        
        # Test anomaly detection
        X_test_scaled = self.feature_scaler.transform(self.X_test)
        anomaly_predictions = self.anomaly_detector.predict(X_test_scaled)
        anomaly_scores = self.anomaly_detector.decision_function(X_test_scaled)
        
        # Count anomalies
        num_anomalies = np.sum(anomaly_predictions == -1)  # -1 = anomaly, 1 = normal
        anomaly_percentage = (num_anomalies / len(anomaly_predictions)) * 100
        
        print(f"\n🔍 Anomaly Detection Results:")
        print(f"   Total test samples: {len(anomaly_predictions)}")
        print(f"   Detected anomalies: {num_anomalies}")
        print(f"   Anomaly rate: {anomaly_percentage:.1f}%")
        
        # Show some examples
        print(f"\n🔬 Example Anomaly Scores (lower = more anomalous):")
        for i in range(min(5, len(anomaly_scores))):
            status = "ANOMALY" if anomaly_predictions[i] == -1 else "NORMAL"
            device_type = self.data_generator.get_device_name(self.y_device_test[i])
            print(f"   {device_type}: {anomaly_scores[i]:.3f} ({status})")
        
        return num_anomalies, anomaly_percentage
    
    def step_4_build_risk_predictor(self):
        """Step 4: Build risk prediction model"""
        print("\n🎯 STEP 4: RISK PREDICTION MODEL")
        print("=" * 50)
        
        print("⚡ Building Random Forest Regressor for risk prediction...")
        print("   Regression = Predicting continuous numbers (0.0 to 1.0 risk)")
        print("   Classification = Predicting categories (device types)")
        
        # Create the risk predictor
        self.risk_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        print(f"   🌲 Forest size: {self.risk_predictor.n_estimators} trees")
        
        # Train the model
        X_train_scaled = self.feature_scaler.transform(self.X_train)
        X_test_scaled = self.feature_scaler.transform(self.X_test)
        
        print("🎓 Training risk predictor...")
        self.risk_predictor.fit(X_train_scaled, self.y_risk_train)
        print("✅ Risk predictor training complete!")
        
        # Test the model
        risk_predictions = self.risk_predictor.predict(X_test_scaled)
        
        # Calculate error metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(self.y_risk_test, risk_predictions)
        mae = mean_absolute_error(self.y_risk_test, risk_predictions)
        r2 = r2_score(self.y_risk_test, risk_predictions)
        
        print(f"\n📊 Risk Prediction Performance:")
        print(f"   Mean Absolute Error: {mae:.3f}")
        print(f"   Mean Squared Error: {mse:.3f}")
        print(f"   R² Score: {r2:.3f} (closer to 1.0 = better)")
        
        # Show some examples
        print(f"\n🔬 Example Risk Predictions:")
        for i in range(min(5, len(risk_predictions))):
            actual = self.y_risk_test[i]
            predicted = risk_predictions[i]
            device_type = self.data_generator.get_device_name(self.y_device_test[i])
            print(f"   {device_type}: Actual={actual:.3f}, Predicted={predicted:.3f}")
        
        return mae, r2
    
    def step_5_test_on_new_data(self):
        """Step 5: Test all models on completely new data"""
        print("\n🎯 STEP 5: TESTING ON NEW DATA")
        print("=" * 50)
        
        # Create some test cases
        test_cases = [
            {
                "name": "Typical Web Server",
                "ports": [22, 80, 443],
                "expected": "Should be classified as Linux Server, low-medium risk"
            },
            {
                "name": "Suspicious Device",
                "ports": [23, 21, 3389, 445, 135],
                "expected": "Should be high risk, possibly anomalous"
            },
            {
                "name": "Simple IoT Device",
                "ports": [80],
                "expected": "Should be IoT device, low risk"
            },
            {
                "name": "Unusual Device",
                "ports": [1337, 31337, 8080, 9999],
                "expected": "Should be anomalous, unknown risk"
            }
        ]
        
        print("🧪 Testing models on new, realistic scenarios...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['name']} ---")
            print(f"Open ports: {test_case['ports']}")
            print(f"Expected: {test_case['expected']}")
            
            # Extract features
            features = self.data_generator.extract_features(test_case['ports'])
            features_array = np.array([features])  # Shape: (1, 10)
            features_scaled = self.feature_scaler.transform(features_array)
            
            # Run all three models
            # 1. Device Classification
            device_pred = self.device_classifier.predict(features_scaled)[0]
            device_proba = self.device_classifier.predict_proba(features_scaled)[0]
            device_confidence = max(device_proba)
            device_name = self.data_generator.get_device_name(device_pred)
            
            # 2. Risk Prediction
            risk_score = self.risk_predictor.predict(features_scaled)[0]
            
            # 3. Anomaly Detection
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = anomaly_score < -0.1  # Threshold for anomaly
            
            print(f"🤖 AI Analysis Results:")
            print(f"   Device Type: {device_name} (confidence: {device_confidence:.3f})")
            print(f"   Risk Score: {risk_score:.3f} ({'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.4 else 'LOW'})")
            print(f"   Anomaly Status: {'ANOMALOUS' if is_anomaly else 'NORMAL'} (score: {anomaly_score:.3f})")
    
    def run_complete_tutorial(self):
        """Run the complete tutorial"""
        print("🎓 SCIKIT-LEARN NETWORK SECURITY TUTORIAL")
        print("🤖 Learn ML by Building Real AI Models!")
        print("=" * 60)
        
        # Step 1: Data
        self.step_1_generate_and_explore_data()
        
        # Step 2: Classification
        accuracy = self.step_2_build_device_classifier()
        
        # Step 3: Anomaly Detection
        num_anomalies, anomaly_rate = self.step_3_build_anomaly_detector()
        
        # Step 4: Regression
        mae, r2 = self.step_4_build_risk_predictor()
        
        # Step 5: Real-world testing
        self.step_5_test_on_new_data()
        
        # Summary
        print("\n🏆 TUTORIAL COMPLETE!")
        print("=" * 50)
        print("🎯 What you've learned:")
        print(f"   ✅ Random Forest Classification (accuracy: {accuracy:.3f})")
        print(f"   ✅ Isolation Forest Anomaly Detection ({anomaly_rate:.1f}% anomalies)")
        print(f"   ✅ Random Forest Regression (R²: {r2:.3f})")
        print(f"   ✅ Feature Engineering and Data Preprocessing")
        print(f"   ✅ Model Training, Testing, and Evaluation")
        
        print("\n🚀 Next steps:")
        print("   📚 Try different algorithms (SVM, Neural Networks, etc.)")
        print("   🔧 Experiment with feature engineering")
        print("   📊 Add data visualization with matplotlib")
        print("   🌐 Apply to your own datasets!")

# Run the tutorial
if __name__ == "__main__":
    tutorial = NetworkSecurityMLTutorial()
    tutorial.run_complete_tutorial()
```

---

## 📊 **Part 3: Advanced Analysis and Visualization**

### **Step 3: Create the Model Evaluator** (`model_evaluator.py`)

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, validation_curve
import pandas as pd

class ModelEvaluator:
    """Advanced model evaluation and visualization"""
    
    def __init__(self, tutorial_instance):
        self.tutorial = tutorial_instance
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup matplotlib for pretty plots"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self):
        """Visualize classification results"""
        print("📊 Creating Confusion Matrix...")
        
        # Get predictions
        X_test_scaled = self.tutorial.feature_scaler.transform(self.tutorial.X_test)
        predictions = self.tutorial.device_classifier.predict(X_test_scaled)
        
        # Create confusion matrix
        cm = confusion_matrix(self.tutorial.y_device_test, predictions)
        device_names = [self.tutorial.data_generator.get_device_name(i) for i in range(6)]
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=device_names, yticklabels=device_names)
        plt.title('Device Classification Confusion Matrix')
        plt.ylabel('Actual Device Type')
        plt.xlabel('Predicted Device Type')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """Visualize which features are most important"""
        print("🔥 Plotting Feature Importance...")
        
        feature_names = [
            "Total Ports", "Has SSH", "Has HTTP", "Has HTTPS", 
            "Has Telnet", "Has RDP", "Has SMB", "Has FTP",
            "Port Spread", "High Ports"
        ]
        
        importances = self.tutorial.device_classifier.feature_importances_
        
        # Create plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]  # Sort by importance
        
        plt.bar(range(len(importances)), importances[indices])
        plt.title('Feature Importance in Device Classification')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_risk_predictions(self):
        """Visualize risk prediction accuracy"""
        print("⚡ Plotting Risk Prediction Results...")
        
        X_test_scaled = self.tutorial.feature_scaler.transform(self.tutorial.X_test)
        risk_predictions = self.tutorial.risk_predictor.predict(X_test_scaled)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.tutorial.y_risk_test, risk_predictions, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)  # Perfect prediction line
        plt.xlabel('Actual Risk Score')
        plt.ylabel('Predicted Risk Score')
        plt.title('Risk Prediction Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self):
        """Show how model performance improves with more data"""
        print("📈 Creating Learning Curves...")
        
        X_train_scaled = self.tutorial.feature_scaler.transform(self.tutorial.X_train)
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.tutorial.device_classifier, X_train_scaled, self.tutorial.y_device_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves - Device Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_comprehensive_report(self):
        """Create a complete analysis report"""
        print("\n📋 COMPREHENSIVE MODEL ANALYSIS REPORT")
        print("=" * 60)
        
        # Run all visualizations
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.plot_risk_predictions()
        self.plot_learning_curves()
        
        print("✅ All visualizations complete!")
        print("💡 Key insights to look for:")
        print("   🎯 Confusion Matrix: Which device types get confused?")
        print("   🔥 Feature Importance: Which network patterns matter most?")
        print("   ⚡ Risk Predictions: How accurate are our risk assessments?")
        print("   📈 Learning Curves: Do we need more training data?")

# Usage example
if __name__ == "__main__":
    from model_trainer import NetworkSecurityMLTutorial
    
    # Run the tutorial first
    tutorial = NetworkSecurityMLTutorial()
    tutorial.run_complete_tutorial()
    
    # Then create advanced analysis
    evaluator = ModelEvaluator(tutorial)
    evaluator.create_comprehensive_report()
```

---

## 🎯 **Part 4: Complete Working Example**

### **Step 4: Create the Main Application** (`network_security_ml.py`)

```python
#!/usr/bin/env python3
"""
🎓 Complete Scikit-Learn Tutorial: Network Security AI
Learn machine learning by building real security models!
"""

from data_generator import NetworkDataGenerator
from model_trainer import NetworkSecurityMLTutorial
from model_evaluator import ModelEvaluator
import argparse

def main():
    """Main tutorial application"""
    parser = argparse.ArgumentParser(description='Scikit-Learn Network Security Tutorial')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'interactive'], 
                       default='basic', help='Tutorial mode')
    parser.add_argument('--samples', type=int, default=100, 
                       help='Samples per device type (default: 100)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Show visualizations (requires matplotlib)')
    
    args = parser.parse_args()
    
    print("🎓 SCIKIT-LEARN NETWORK SECURITY TUTORIAL")
    print("🤖 Learn AI by Building Real Security Models!")
    print("=" * 60)
    
    if args.mode == 'basic':
        # Run basic tutorial
        tutorial = NetworkSecurityMLTutorial()
        tutorial.run_complete_tutorial()
        
    elif args.mode == 'advanced':
        # Run with advanced analysis
        tutorial = NetworkSecurityMLTutorial()
        tutorial.run_complete_tutorial()
        
        if args.visualize:
            evaluator = ModelEvaluator(tutorial)
            evaluator.create_comprehensive_report()
            
    elif args.mode == 'interactive':
        # Interactive mode
        run_interactive_tutorial()

def run_interactive_tutorial():
    """Interactive tutorial mode"""
    print("\n🎮 INTERACTIVE MODE")
    print("Let's build models step by step!")
    
    tutorial = NetworkSecurityMLTutorial()
    
    input("Press Enter to start with data generation...")
    tutorial.step_1_generate_and_explore_data()
    
    input("\nPress Enter to build device classifier...")
    tutorial.step_2_build_device_classifier()
    
    input("\nPress Enter to build anomaly detector...")
    tutorial.step_3_build_anomaly_detector()
    
    input("\nPress Enter to build risk predictor...")
    tutorial.step_4_build_risk_predictor()
    
    input("\nPress Enter to test on new data...")
    tutorial.step_5_test_on_new_data()
    
    print("\n🏆 Interactive tutorial complete!")
    
    # Let user test their own data
    while True:
        print("\n🧪 Test your own network data!")
        print("Enter port numbers separated by commas (or 'quit' to exit):")
        user_input = input("Ports: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        try:
            ports = [int(p.strip()) for p in user_input.split(',')]
            test_custom_device(tutorial, ports)
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")

def test_custom_device(tutorial, ports):
    """Test user's custom device"""
    print(f"\n🔍 Analyzing device with ports: {ports}")
    
    # Extract features
    features = tutorial.data_generator.extract_features(ports)
    features_array = np.array([features])
    features_scaled = tutorial.feature_scaler.transform(features_array)
    
    # Run all models
    device_pred = tutorial.device_classifier.predict(features_scaled)[0]
    device_name = tutorial.data_generator.get_device_name(device_pred)
    device_confidence = max(tutorial.device_classifier.predict_proba(features_scaled)[0])
    
    risk_score = tutorial.risk_predictor.predict(features_scaled)[0]
    
    anomaly_score = tutorial.anomaly_detector.decision_function(features_scaled)[0]
    is_anomaly = anomaly_score < -0.1
    
    print(f"🤖 AI Analysis:")
    print(f"   Device Type: {device_name} (confidence: {device_confidence:.3f})")
    print(f"   Risk Level: {risk_score:.3f} ({'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.4 else 'LOW'})")
    print(f"   Anomaly: {'YES' if is_anomaly else 'NO'} (score: {anomaly_score:.3f})")

if __name__ == "__main__":
    main()
```

---

## 🚀 **Running the Tutorial**

### **Basic Tutorial**
```bash
python network_security_ml.py --mode basic
```

### **Advanced with Visualizations**
```bash
python network_security_ml.py --mode advanced --visualize
```

### **Interactive Learning**
```bash
python network_security_ml.py --mode interactive
```

### **Custom Dataset Size**
```bash
python network_security_ml.py --samples 200  # More training data
```

---

## 🎓 **Key Learning Outcomes**

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

---

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

### **2. Feature Engineering**
```python
def enhanced_feature_extraction(ports):
    """Add more sophisticated features"""
    features = basic_feature_extraction(ports)
    
    # Add new features
    features.extend([
        len([p for p in ports if p < 1024]),  # System ports
        1 if any(p in [80, 8080, 443, 8443] for p in ports) else 0,  # Web services
        1 if any(p in [21, 22, 23, 3389] for p in ports) else 0,     # Remote access
    ])
    
    return features
```

### **3. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Test model with cross-validation
cv_scores = cross_val_score(classifier, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f}")
```

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
