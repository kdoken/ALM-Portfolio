# ALM-Portfolio App

This project is a Python and Streamlit-based Asset–Liability Management (ALM) application built as an exercise in financial mathematics, quantitative modeling, and software development.  
It was designed to strengthen technical skills and practical understanding of financial analysis while building a functional tool that can be used for real-world portfolio management and actuarial applications.

---

## Overview

The ALM Portfolio Analyzer allows users to construct, analyze, and visualize financial portfolios composed of various types of cash-flow instruments.  
It supports both **structured** (e.g., annuities, lump sums) and **custom** cash flows, flexible interest rate models, and standard ALM risk metrics such as duration and sensitivity.

---

## Features

- **Portfolio Builder** – Add structured or custom instruments  
  - Lump Sum, Annuity, and Annuity + Final Payment  
- **Interest Rate Models** – Effective, Simple, Nominal, Force, Delta Function, and Rate Schedule  
- **Duration Metrics** – Present Value (PV), Macaulay Duration, Modified Duration, PV01, and Effective Duration  
- **Custom Cash Flows** – Define irregular or user-specified cash-flow patterns  
- **Automatic Storage** – All portfolio entries are saved to `portfolio.csv` for later review or analysis  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kdoken/ALM-Portfolio.git
   cd ALM-Portfolio
Create and activate a virtual environment:

bash
Copy code
python -m venv .venv
source .venv/bin/activate   - macOS/Linux
.venv\Scripts\activate    - Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
To launch the Streamlit interface:

bash
Copy code
streamlit run ui.py
Then open the provided local URL (typically http://localhost:8501) in your browser.

##Project Structure
graphql
Copy code
ALM-Portfolio/
├── ui.py                 # Streamlit user interface
├── functions.py          # Core ALM analysis functions
├── util.py               # Utility and JSON-safe parsing functions
├── portfolio.csv         # Saved portfolio data
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

##Example Capabilities
Construct portfolios containing multiple asset and liability types

Evaluate duration mismatches and rate sensitivity for ALM studies

Simulate interest rate shocks (e.g., ±25 bp parallel shifts)

Explore rate models with time-dependent delta functions

Serve as a self-contained environment for testing actuarial and financial concepts

##Purpose
This project was developed as part of my continued learning in financial mathematics, quantitative analysis, and actuarial modeling.
It serves both as a learning exercise and as a technical demonstration for employers and collaborators—illustrating skills in Python, Streamlit, data analysis, and applied finance.

##Author
K. Doken
Double Major in Physics & Applied Mathematics – UC Berkeley
Focused on actuarial, quantitative, and data-driven financial modeling
GitHub: github.com/kdoken
