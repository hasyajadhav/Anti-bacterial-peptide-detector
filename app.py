import streamlit as st
#-----------------------------------------------FEATURES--------------------------------
from collections import Counter

def AAC(seq):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    count = Counter(seq)
    total = len(seq) or 1

    return {f"AAC_{a}": count.get(a, 0)/total for a in aa}
import itertools

def DPC(seq):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    
    # Generate all 400 dipeptides
    dipeptides = [''.join(p) for p in itertools.product(aa, repeat=2)]
    
    # Initialize all to 0
    count = {f"DPC_{dp}": 0 for dp in dipeptides}

    # Count observed dipeptides
    for i in range(len(seq) - 1):
        dp = seq[i:i+2]
        key = f"DPC_{dp}"
        if key in count:
            count[key] += 1

    # Normalize
    total = sum(count.values()) or 1
    return {k: v / total for k, v in count.items()}
def ATC(seq):
    atom_count = {"C":0, "H":0, "N":0, "O":0, "S":0}

    aa_atoms = {
        'A': {'C':3,'H':7,'N':1,'O':2,'S':0},
        'C': {'C':3,'H':7,'N':1,'O':2,'S':1},
        'D': {'C':4,'H':7,'N':1,'O':4,'S':0},
        'E': {'C':5,'H':9,'N':1,'O':4,'S':0},
        'F': {'C':9,'H':11,'N':1,'O':2,'S':0},
        'G': {'C':2,'H':5,'N':1,'O':2,'S':0},
        'H': {'C':6,'H':9,'N':3,'O':2,'S':0},
        'I': {'C':6,'H':13,'N':1,'O':2,'S':0},
        'K': {'C':6,'H':14,'N':2,'O':2,'S':0},
        'L': {'C':6,'H':13,'N':1,'O':2,'S':0},
        'M': {'C':5,'H':11,'N':1,'O':2,'S':1},
        'N': {'C':4,'H':8,'N':2,'O':3,'S':0},
        'P': {'C':5,'H':9,'N':1,'O':2,'S':0},
        'Q': {'C':5,'H':10,'N':2,'O':3,'S':0},
        'R': {'C':6,'H':14,'N':4,'O':2,'S':0},
        'S': {'C':3,'H':7,'N':1,'O':3,'S':0},
        'T': {'C':4,'H':9,'N':1,'O':3,'S':0},
        'V': {'C':5,'H':11,'N':1,'O':2,'S':0},
        'W': {'C':11,'H':12,'N':2,'O':2,'S':0},
        'Y': {'C':9,'H':11,'N':1,'O':3,'S':0},
    }

    # Sum atoms
    for aa in seq:
        if aa in aa_atoms:
            for atom in atom_count:
                atom_count[atom] += aa_atoms[aa][atom]

    # Normalize
    total = sum(atom_count.values()) or 1
    return {f"ATC_{k}": v/total for k, v in atom_count.items()}
import itertools

def CTC(seq):
    groups = {
        'A':'1','G':'1','V':'1',
        'I':'2','L':'2','F':'2','P':'2',
        'Y':'3','M':'3','T':'3','S':'3',
        'H':'4','N':'4','Q':'4','W':'4',
        'R':'5','K':'5',
        'D':'6','E':'6',
        'C':'7'
    }

    # Convert sequence → grouped numbers
    grouped = ''.join([groups.get(a, '0') for a in seq])

    #  Create ALL possible triads (fixed feature space)
    triads = [''.join(p) for p in itertools.product('1234567', repeat=3)]

    # Initialize all to 0
    count = {f"CTC_{tri}": 0 for tri in triads}

    # Count occurrences
    for i in range(len(grouped)-2):
        tri = grouped[i:i+3]
        key = f"CTC_{tri}"
        if key in count:
            count[key] += 1

    # Normalize
    total = sum(count.values()) or 1
    return {k: v/total for k, v in count.items()}
def RRI(seq):
    max_repeat = 1
    current = 1

    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current += 1
            max_repeat = max(max_repeat, current)
        else:
            current = 1

    return {"RRI_max_repeat": max_repeat}
def DDR(seq):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    result = {}

    for aa in amino_acids:
        positions = [i for i, x in enumerate(seq) if x == aa]

        if not positions:
            result[f"DDR_{aa}"] = 0
            continue

        # Reverse positions (for end distance)
        rev_positions = [i for i, x in enumerate(seq[::-1]) if x == aa]

        gaps = []

        # gaps between occurrences
        for i in range(len(positions) - 1):
            gaps.append(positions[i+1] - positions[i] - 1)

        # add start + end gaps
        gaps.insert(0, positions[0])
        gaps.append(rev_positions[0])

        # compute DDR
        if gaps:
            numerator = sum(g*g for g in gaps)
            denominator = sum(gaps) + 1
            result[f"DDR_{aa}"] = numerator / denominator
        else:
            result[f"DDR_{aa}"] = 0

    return result
def PCP(seq):
    total = len(seq) or 1

    hydrophobic = "AILMFWYV"
    polar = "STNQ"
    positive = "KRH"
    negative = "DE"

    pos_count = sum(1 for a in seq if a in positive)
    neg_count = sum(1 for a in seq if a in negative)

    return {
        "PCP_hydrophobic": sum(1 for a in seq if a in hydrophobic)/total,
        "PCP_polar": sum(1 for a in seq if a in polar)/total,
        "PCP_positive": pos_count/total,
        "PCP_negative": neg_count/total,
        "PCP_net_charge": (pos_count - neg_count)/total
    }
def SEP(seq):
    from collections import Counter
    import math

    count = Counter(seq)
    total = len(seq) or 1

    entropy = -sum((c/total) * math.log2(c/total) for c in count.values())
    return {"SEP_entropy": entropy}
def ALL_FEATURES(seq):
    features = {}

    features.update(AAC(seq))
    features.update(DPC(seq))
    features.update(ATC(seq))
    features.update(CTC(seq))
    features.update(RRI(seq))
    features.update(DDR(seq))
    features.update(PCP(seq))
    features.update(SEP(seq))

    

    return features
#--------------------------------------------LOAD MODELS---------------------------------------------
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("neg_model_selected.pkl")
scaler = joblib.load("neg_scaler_selected.pkl")
selected_features = joblib.load("neg_selected_features.pkl")

model_P = joblib.load("abp_model.pkl")
scaler_P = joblib.load("abp_scaler.pkl")
selected_P=joblib.load("sel_feature.pkl")


##-------------------------------------------------------------------------------------



# 1. Page Configuration
st.set_page_config(
    page_title='ANTI-BACTERIAL PEPTIDES',
    layout="centered"
)

# 2. Sidebar Implementation
# This creates the collapsible sidebar with the arrow button

with st.sidebar:
    st.header("About This Tool") # Removed the ###, st.header already makes it a header
    st.write("""
    This tool is designed to provide rapid identification of **Anti-Bacterial Peptides**.
    By analyzing sequence patterns, the underlying model helps researchers screen potential
    antimicrobial candidates efficiently. This is particularly useful in the early
    stages of drug discovery and bioinformatics research.
    """) # <--- Ensure this looks exactly like this

    st.header("Home")
    st.write("""
    **Contact:**
    * Hasya Jadhav : 3522511027@despu.edu.in
    * Madhura Bhubihar : 3522511012@despu.edu.in
    """)
    
    st.divider() # Optional: adds a visual line
    st.caption(" Bioinformatics tool")

# 3. Main Page Title
st.title("🧬 ANTI-BACTERIAL PEPTIDES")

seq_input = st.text_area("Enter Peptide Sequence")
valid = set("ACDEFGHIKLMNPQRSTVWY")
target=st.selectbox("Select bacterial type",["Gram positive","Gram negative"])
perdict=st.button("predict")

if perdict:
    if not set(seq_input).issubset(valid):
        st.error("Invalid sequence: contains non-amino acid character")
    else:
        st.success("Processing single sequence..")
        if target=="Gram positive":
            features=ALL_FEATURES(seq_input)
            df_1=pd.DataFrame([features], columns=features)
            df_sel=df_1[selected_P]
            scaled_p=scaler_P.transform(df_sel)
            prob=model_P.predict_proba(scaled_p)[0][1]
            
            if prob > 0.8:
                st.success("🟢 Strongly Antibacterial")
            elif prob > 0.6:
                st.info("🟡 Moderately Antibacterial")
            else:
                st.error("🔴 Not Antibacterial")
            st.metric(label="Prediction Confidence", value=f"{prob*100:.2f}%")
            color = "green" if prob > 0.5 else "red"       
        else:
            features=ALL_FEATURES(seq_input)
            df = pd.DataFrame([features], columns=features)
            df_selected = df[selected_features]
            scaled = scaler.transform(df_selected)
            prob = model.predict_proba(scaled)[0][1]
#------------------answer------------------------------------------------
            if prob > 0.8:
                st.success("🟢 Strongly Antibacterial")
            elif prob > 0.6:
                st.info("🟡 Moderately Antibacterial")
            else:
                st.error("🔴 Not Antibacterial")
            st.metric(label="Prediction Confidence", value=f"{prob*100:.2f}%")
            color = "green" if prob > 0.5 else "red"
            

        

          

            
