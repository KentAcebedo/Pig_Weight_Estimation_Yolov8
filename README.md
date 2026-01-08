# Pig Weight Estimation

An automated system for estimating the weight of live swine using the **YOLOv8** algorithm. This project leverages computer vision to provide a non-invasive, efficient alternative to manual weighing.

---

## 📌 Project Overview
Manual weighing can be stressful for livestock and labor-intensive for farmers. This system automates the process by detecting a pig's body dimensions via YOLOv8 and calculating the estimated weight using standard biometric formulas.

## 🧪 Methodology
The weight estimation is calculated using the following formula:

$$Weight = \frac{HG^2 \times L}{400}$$

Where:
* **HG**: Heart Girth
* **L**: Body Length

The YOLOv8 model is used to accurately identify and measure these physical parameters from image or video data.

## 📊 Results & Validation
The model's performance was statistically validated against manual measurements:

* **Average Predicted Weight:** 111.8 kg
* **Actual Average Weight:** 111.1 kg
* **Statistical Evaluation:** An independent **t-test** revealed no significant differences between estimated and actual weights, confirming the model's effectiveness and accuracy.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.8+
* YOLOv8 (Ultralytics)

