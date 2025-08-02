# Classification vs Regression

**Two different ways to make predictions**

## What's the difference?

When you're building a machine learning model, you're basically teaching a computer to make predictions. The type of prediction you want determines whether you use classification or regression.

**Classification**: "What category is this thing?"
- Puts things into buckets or categories
- Example: Is this email spam or not spam?

**Regression**: "What number will this be?"
- Predicts actual numerical values
- Example: How much will this house cost?

## Pizza ordering example

Let's say you're ordering pizza online:

**Classification questions:**
- What type of pizza is this? → Pepperoni, Margherita, or Hawaiian
- Is this pizza good or bad? → Good or Bad  
- What size is this pizza? → Small, Medium, or Large

**Regression questions:**
- How much will this pizza cost? → $12.99
- How many calories are in this slice? → 285 calories
- How long until delivery? → 23.5 minutes

See the difference? Classification gives you categories, regression gives you numbers.

## Quick comparison

| | Classification | Regression |
|---|---|---|
| **Output** | Categories/Labels | Numbers |
| **Examples** | "Cat", "Spam", "Fraud" | 15.7, $299, 0.85 |
| **Goal** | Sort things into boxes | Predict exact values |
| **Answers** | Discrete choices | Any number in a range |
## House buying example

Same house, different questions:

**Classification: "What type of house is this?"**
```
Input: [3 bedrooms, 2 bathrooms, 1500 sq ft, garage]
Output: "Ranch Style" (could also be Colonial, Victorian, etc.)
```

**Regression: "How much is this house worth?"**
```
Input: [3 bedrooms, 2 bathrooms, 1500 sq ft, garage]  
Output: $425,000 (could be any dollar amount)
```

## Network security examples

In our tutorial, we use both approaches:

**Classification: Device type detection**
```python
# Input: Network ports [22, 80, 443, 3389]
# Output: "Windows Server" (could be Linux Server, IoT Device, etc.)

device_types = [
    "IoT Device",      
    "Linux Server",    
    "Windows Server",  
    "Personal Laptop", 
    "Router",          
    "Suspicious"       
]
```

**Regression: Risk score prediction**
```python
# Input: Network ports [22, 80, 443, 3389]
# Output: 0.73 (risk score between 0.0 and 1.0)
```

## How to choose?

Ask yourself: "What kind of answer do I want?"

**Use classification when:**
- You want to sort or categorize things
- You have predefined groups or labels
- The answer is "this OR that"
- Examples: spam detection, image recognition, fraud detection

**Use regression when:**
- You want to predict a number
- The answer can be any value in a range
- You're estimating quantities
- Examples: price prediction, weather forecasting, risk scoring

## Common algorithms

**Classification algorithms:**
- Random Forest Classifier (what we use!)
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree

**Regression algorithms:**
- Random Forest Regressor (what we use!)
- Linear Regression
- Support Vector Regression
- Decision Tree Regressor
- Neural Networks

## Tips for success

**Classification tips:**
- Make sure you have roughly equal examples of each category
- Use confusion matrices to see which categories get mixed up
- Accuracy = percentage of correct predictions

**Regression tips:**
- Make sure all your input numbers are on similar scales
- Track your average error (how far off your predictions are)
- R-squared tells you how well your model explains the data (1.0 = perfect)

## Test your understanding

Try to identify each scenario:

1. Predicting if a credit card transaction is fraudulent or legitimate
   - Answer: Classification (Fraud vs Legitimate)

2. Estimating how much rainfall there will be tomorrow
   - Answer: Regression (Number in inches/millimeters)

3. Determining if a medical scan shows cancer or no cancer
   - Answer: Classification (Cancer vs No Cancer)

4. Predicting the selling price of a used car
   - Answer: Regression (Dollar amount)

## What's next?

Now that you get the difference, try the main tutorial. You'll build both classification and regression models for network security.

Remember: Classification sorts things into boxes, regression predicts numbers. When in doubt, ask yourself "Am I trying to categorize something or estimate a quantity?"
