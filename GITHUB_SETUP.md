# GitHub Repository Setup Guide

## Quick Setup (5 minutes)

### Step 1: Initialize Git Repository

Open PowerShell in your project directory and run:

```powershell
cd D:\Studies\DDP

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete Deep Hedging implementation"
```

### Step 2: Create .gitignore

Create `.gitignore` file to exclude unnecessary files:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt
checkpoints/

# Environment
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data
data/
*.csv
*.h5
```

### Step 3: Create GitHub Repository

1. **Go to GitHub**: https://github.com
2. **Click** "New repository" (green button)
3. **Fill in**:
   - Repository name: `deep-hedging`
   - Description: "Production-ready Deep Hedging implementation for derivative hedging"
   - Visibility: Public or Private (your choice)
   - **DO NOT** initialize with README (you already have one)
4. **Click** "Create repository"

### Step 4: Connect Local to GitHub

GitHub will show you commands. Use these:

```powershell
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/deep-hedging.git

# Push to GitHub
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 5: Verify Upload

Go to `https://github.com/YOUR_USERNAME/deep-hedging` and verify all files are there.

---

## Alternative: Using GitHub Desktop (Easier)

### Step 1: Download GitHub Desktop
- Download from: https://desktop.github.com/
- Install and sign in

### Step 2: Add Repository
1. Click "File" → "Add local repository"
2. Choose `D:\Studies\DDP`
3. Click "Create repository" if prompted

### Step 3: Publish to GitHub
1. Click "Publish repository"
2. Name: `deep-hedging`
3. Description: "Production-ready Deep Hedging implementation"
4. Click "Publish repository"

Done! ✅

---

## What to Include in Repository

### Essential Files (Already created):
- ✅ Source code (`deep_hedging/`)
- ✅ Documentation (`README.md`, `docs/`)
- ✅ Setup files (`setup.py`, `requirements.txt`)
- ✅ Usage guide (`USAGE_GUIDE.md`)

### Optional (Exclude from Git):
- ❌ Trained models (`checkpoints/`)
- ❌ Large datasets
- ❌ Jupyter checkpoint files

---

## Repository Structure on GitHub

```
deep-hedging/
├── deep_hedging/          # Source code
├── docs/                  # Documentation
├── README.md              # Project overview
├── USAGE_GUIDE.md         # How to use
├── setup.py               # Installation
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── LICENSE               # License file (optional)
```

---

## Adding a License (Optional but Recommended)

### MIT License (Most permissive):

Create `LICENSE` file:

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Updating Repository Later

When you make changes:

```powershell
# Check status
git status

# Add changes
git add .

# Commit with message
git commit -m "Add new feature: XYZ"

# Push to GitHub
git push
```

---

## Common Git Commands

```powershell
# Check status
git status

# View commit history
git log --oneline

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main

# Pull latest changes
git pull

# View remote URL
git remote -v
```

---

## Troubleshooting

### Issue: "fatal: not a git repository"
**Solution**: Run `git init` first

### Issue: "Permission denied (publickey)"
**Solution**: Use HTTPS instead of SSH, or set up SSH keys

### Issue: "Updates were rejected"
**Solution**: Pull first, then push:
```powershell
git pull origin main --rebase
git push
```

---

**Your repository is now live on GitHub! 🎉**
