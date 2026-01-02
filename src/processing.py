import pandas as pd
import numpy as np
import re
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from typing import cast

def clean_mtg_stat(val):
    """
    Parses MTG stats (Power/Toughness/Loyalty/Defense) including split cards (//) 
    and variable values (*).
    """
    if pd.isna(val) or val in ['{n}', 'n', '']:
        return 0, 0, 0, 0 # main, side, is_variable, is_applicable

    clean_val = str(val).replace('{', '').replace('}', '').strip()
    is_variable = 1 if '*' in clean_val else 0
    
    # Handle split cards
    if '//' in clean_val:
        parts = clean_val.split('//')
        f_str, b_str = parts[0].strip(), parts[1].strip()
    else:
        f_str, b_str = clean_val, 'n'

    def extract_num(s):
        if not s or s == 'n': return 0
        match = re.search(r'-?\d+', s)
        return int(match.group()) if match else 0

    front_app = 1 if ('n' not in f_str and f_str != '') else 0
    back_app  = 1 if ('n' not in b_str and b_str != '') else 0
    
    return extract_num(f_str), extract_num(b_str), is_variable, max(front_app, back_app)

def parse_keywords(val):
    """Parses the keywords string column into a Python list."""
    if pd.isna(val) or val == '[]':
        return []
    try:
        # Safe evaluation of string representation of lists
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def mask_self_reference(row):
    """
    Replaces the card's name in its oracle text with [SELF] to prevent 
    the model from biasing similarity based on card names.
    """
    name = str(row['name'])
    text = str(row['oracle_text'])
    
    if name and text and text != 'nan':
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        return pattern.sub('[SELF]', text)
    return text

def preprocess_dataframe(df):
    """
    Wrapper function to apply all cleaning steps to the dataframe.
    """
    # Process Power and Toughness
    df[['p_main', 'p_side', 'p_var', 'p_exists']] = df['power'].apply(
        lambda x: pd.Series(clean_mtg_stat(x))
    )
    df[['t_main', 't_side', 't_var', 't_exists']] = df['toughness'].apply(
        lambda x: pd.Series(clean_mtg_stat(x))
    )
    
    # Process Keywords
    df['keyword_list'] = df['keywords'].apply(parse_keywords)
    
    # Process Types
    df['types_list'] = df['type_line'].str.split(r'[^a-zA-Z0-9]+').apply(
        lambda x: [i for i in x if i] if isinstance(x, list) else []
    )
    
    # Apply Masking
    df['masked_text'] = df.apply(mask_self_reference, axis=1)
    
    return df

def encode_categorical_features(df: pd.DataFrame):
    """
    Transforms list-based columns into binary indicator variables.
    Explicitly converts to dense array to satisfy Pylance's spmatrix error.
    """
    # 1. Process Types
    mlb_type = MultiLabelBinarizer(sparse_output=False) # Ensure dense output
    # Force the output to a dense numpy array to resolve spmatrix issues
    type_data = cast(np.ndarray, mlb_type.fit_transform(df['types_list']))
    
    type_dummies = pd.DataFrame(data=type_data, index=df.index)
    type_dummies.columns = [str(c) for c in mlb_type.classes_]
    type_dummies = type_dummies.add_prefix('type_')

    # 2. Process Keywords
    mlb_kw = MultiLabelBinarizer(sparse_output=False)
    kw_data = cast(np.ndarray, mlb_kw.fit_transform(df['keyword_list']))
    
    keyword_dummies = pd.DataFrame(data=kw_data, index=df.index)
    keyword_dummies.columns = [str(c) for c in mlb_kw.classes_]
    keyword_dummies = keyword_dummies.add_prefix('kw_')

    return type_dummies, keyword_dummies