# Diabetes Classification - ML Project

**Team:** aiserpants  
**Branch:** Adrian's`
**Goal:** Build a machine learning model to predict diabetes and deploy it as a web application.

---

## 📅 Timeline (4 Weeks)

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Setup + EDA | Data cleaning, visualizations, baseline understanding |
| **Week 2** | Baseline Models | Logistic Regression, Decision Trees, Random Forest |
| **Week 3** | Advanced Models | XGBoost, Neural Networks, model comparison |
| **Week 4** | Deployment | Streamlit app, documentation, final presentation |

---

## Quick Start

### 1️⃣ Get Access to the Repository

**Sruti will add you as a collaborator:**
```
Settings → Collaborators → Add people → [your GitHub username]
```

Check your email for the invitation and accept it.

---

### 2️⃣ Clone the Repository

```bash
# Clone the repo
git clone https://github.com/SrutiElan/diabetes-classification.git
cd diabetes-classification

# Create YOUR branch (use your name!)
git checkout -b feature/yourname-eda
# Example: git checkout -b feature/john-eda

# Verify you're on your branch
git branch
```

---

### 3️⃣ Work in Google Colab

#### **Method A: Open from GitHub (Easiest)**

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File** → **Open Notebook** → **GitHub** tab
3. Enter: `SrutiElan/diabetes-classification`
4. Select **your branch**: `feature/yourname-eda`
5. Open the notebook (e.g., `notebooks/EDA.ipynb`)

#### **Method B: Clone in Colab (More Control)**

Add this to your first Colab cell:

```python
# Clone repo
!git clone https://github.com/SrutiElan/diabetes-classification.git
%cd diabetes-classification

# Configure git
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Switch to your branch
!git checkout feature/yourname-eda
```

---

### 4️⃣ Save Your Work

#### **Option 1: Save from Colab to GitHub Directly**

1. **File** → **Save a copy in GitHub**
2. Repository: `SrutiElan/diabetes-classification`
3. Branch: `feature/yourname-eda`
4. Commit message: "Added correlation analysis to EDA"
5. Click **OK**

#### **Option 2: Use Git Commands in Colab**

Add this cell after your work:

```python
# Check what changed
!git status

# Save your changes
!git add notebooks/EDA.ipynb
!git commit -m "Completed missing value analysis"
!git push origin feature/yourname-eda
```

---

## 🔄 Daily Workflow

**BEFORE you start working each day:**

```bash
# 1. Switch to main branch
git checkout main

# 2. Get latest changes
git pull origin main

# 3. Switch back to YOUR branch
git checkout feature/yourname-eda

# 4. Merge latest changes into your branch
git merge main
```

**AFTER you finish working:**

```bash
# 1. Save your changes
git add notebooks/your-notebook.ipynb
git commit -m "Describe what you did"

# 2. Push to GitHub
git push origin feature/yourname-eda
```

---

## 📝 Git Cheat Sheet

| Command | What It Does |
|---------|-------------|
| `git status` | See what files you changed |
| `git add <file>` | Prepare file to save |
| `git commit -m "message"` | Save changes with description |
| `git push origin <branch>` | Upload to GitHub |
| `git pull origin main` | Download latest changes |
| `git checkout <branch>` | Switch branches |
| `git branch` | See all branches (* = current) |

**Commit Message Examples:**
```bash
✅ "Added missing value visualization"
✅ "Fixed glucose column zero values"
✅ "Trained Random Forest model"

❌ "fixed stuff"
❌ "asdfasdf"
❌ "final version"
```

---

## 🔀 Creating a Pull Request (PR)

When your work is ready to merge into `main`:

### **Step 1: Push Your Branch**
```bash
git push origin feature/yourname-eda
```

### **Step 2: Create PR on GitHub**

1. Go to [github.com/SrutiElan/diabetes-classification](https://github.com/SrutiElan/diabetes-classification)
2. Click **"Compare & pull request"** button (appears after pushing)
3. **Title:** "Added EDA notebook with visualizations"
4. **Description:**
   ```
   ## What I did
   - Cleaned missing values in glucose/BMI columns
   - Created correlation heatmap
   - Analyzed diabetes vs. age distribution
   
   ## Files changed
   - notebooks/EDA.ipynb
   
   ## Screenshots
   [Paste any key visualizations]
   ```
5. Click **"Create pull request"**

### **Step 3: Code Review**

- Sruti (or teammates) will review
- They may request changes (that's normal!)
- Make requested changes in your branch and push again
- PR updates automatically

### **Step 4: Merge**

- Once approved, **Sruti merges** your code
- Delete your branch after merge (keeps repo clean)

---

## ⚠️ Common Issues

### **"Merge Conflict"**

**What happened:** You and someone else edited the same lines.

**Fix:**
```bash
# 1. Git will mark conflicts in your file like this:
<<<<<<< HEAD
Your code
=======
Their code
>>>>>>> main

# 2. Edit file - keep what you want, delete conflict markers
# 3. Save file
# 4. Mark as resolved:
git add notebooks/EDA.ipynb
git commit -m "Resolved merge conflict"
git push origin feature/yourname-eda
```

### **"Behind origin/main"**

**Fix:**
```bash
git pull origin main
git push origin feature/yourname-eda
```

### **Colab Disconnected - Lost Work**

**Prevention:** Add this to your first cell:
```python
# Auto-save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('notebooks/EDA.ipynb', '/content/drive/MyDrive/backup_EDA.ipynb')
```

---

## 📂 Repository Structure

```
diabetes-classification/
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── baseline_models.ipynb  # Logistic Reg, Decision Trees
│   ├── advanced_models.ipynb  # XGBoost, Neural Networks
│   └── deployment.ipynb       # Model selection & prep
├── data/
│   └── pima-indians-diabetes.csv
├── models/
│   └── (saved models go here)
├── app/
│   └── streamlit_app.py       # Web application
├── requirements.txt
└── README.md
```

---

## 🛠️ Local Setup (Nah chat let's keep it on Colab ok)

If you want to work locally instead of Colab:

```bash
# Clone repo
git clone https://github.com/SrutiElan/diabetes-classification.git
cd diabetes-classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

---

## ✅ Best Practices

1. **Commit often** - Save small, logical chunks of work
2. **Pull before push** - Always get latest changes first
3. **Test your code** - Restart kernel & run all cells before committing
4. **Write good commit messages** - Others should understand what you did
5. **Ask for help** - If stuck for >30 minutes, reach out on Slack/Discord

---

## 📚 Resources

- **Dataset:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Git Tutorial:** [GitHub Guides](https://guides.github.com/)
- **Colab Guide:** [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Scikit-learn Docs:** [scikit-learn.org](https://scikit-learn.org/)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

