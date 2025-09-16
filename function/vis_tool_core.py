# Environment Setting 
# using python 3.11.5
import os
import sys
import time
import json
import ast
import dill 
import joblib
import pickle
import ctypes
import subprocess
import warnings
import platform
import importlib
import string
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import warnings
from pydantic import BaseModel, field_validator, ConfigDict 
from packaging.version import Version
from packaging.specifiers import SpecifierSet

# Data Manipulation 
import numpy as np
import pandas as pd
import itertools
from itertools import chain
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Union, Optional
from pydantic import BaseModel, FilePath, ValidationError, validator

# Natural Language Processing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec #phasing out, but useful for quick static embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

# Semantic & Transformer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

# Machine Learning Analysis 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.manifold import TSNE

# Deep Learning
import torch
# Fix for TqdmWarning by using standard tqdm instead of autonotebook
from tqdm import tqdm

# Visualization
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import networkx as nx_lib  # Renamed to avoid conflicts
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.io as pio

# Interactive Dash
import dash
from dash import dcc, html, Input, Output, State, callback_context

# Suppress Warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tqdm')

# Version Checks 
IMPORT_NAME = {
    "python-dotenv": "dotenv",
    # add others only if pip name ≠ import name
    # e.g., "scikit-learn": "sklearn"
}
def _get_version(pkg_key: str):
    if pkg_key == "python":
        return platform.python_version()
    mod_name = IMPORT_NAME.get(pkg_key, pkg_key)
    try:
        mod = importlib.import_module(mod_name)
        v = getattr(mod, "__version__", None)
        if v: return v
        from importlib.metadata import version as _ver
        return _ver(pkg_key)
    except Exception:
        return None

def _ok(ver: str, spec: str) -> bool:
    try:
        return Version(ver) in SpecifierSet(spec)
    except Exception:
        return False

def check_versions(specs: dict) -> bool:
    print(f"\nPython: {platform.python_version()}  ({platform.system()} {platform.release()})")
    all_ok = True
    for pkg, spec in specs.items():
        ver = _get_version(pkg)
        if ver is None:
            print(f"{pkg:<22} ❌ not installed  (required {spec})")
            all_ok = False
        else:
            mark = "✅" if _ok(ver, spec) else "❌"
            print(f"{pkg:<22} {ver:<12} {mark}  required {spec}")
            all_ok = all_ok and (mark == "✅")
    print("\n" + ("✅ Environment looks compatible." if all_ok else "❌ Version check failed."))
    return all_ok
# -------------------------------------------------------------
# Utility Functions

# User can add/update the default stop list 
def manage_stop_list(stop_list_path, default_stop_words):

    """
    Allow user add/update the default stop list 
    """
    
    # If no stop list path is provided, just use defaults
    extra_stop_words = set()
    if stop_list_path and os.path.exists(stop_list_path):
        try:
            with open(stop_list_path, "r") as f:
                extra_stop_words = {line.strip().lower() for line in f if line.strip()}
        except Exception as e:
            print(f"Error loading stopwords from {stop_list_path}: {str(e)}")
            print("Using default stop words only.")
    else:
        print(f"Custom stop list not found at {stop_list_path}. Using default stop words only.")

    # Remove any words that are in WORD_FAMILIES if it exists
    words_to_exclude = set()
    if 'WORD_FAMILIES' in globals():
        for family_name, family_words in WORD_FAMILIES.items():
            for word in family_words:
                words_to_exclude.add(word.lower())
    
    # Filter out words that should be excluded
    filtered_default_stop_words = {word for word in default_stop_words if word.lower() not in words_to_exclude}
    filtered_extra_stop_words = {word for word in extra_stop_words if word.lower() not in words_to_exclude}
    
    full_stop_words = filtered_default_stop_words.union(filtered_extra_stop_words)
    
    # Print a sample of the custom stopwords for verification
    if filtered_extra_stop_words:
        sample = list(filtered_extra_stop_words)[:10]
    
    return full_stop_words

# Maps a POS tag (like 'NN', 'VB') to WordNet format (like NOUN, VERB).
def get_wordnet_pos(tag):

    """Convert POS tag to WordNet format"""

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Normalize edge weights for prettier and clearer network visuals.
def normalize_edge_weights(weights, scale, base):
    """
    Normalize edge weights for visualization with improved scaling
    
    Parameters:
    -----------
    weights : list
        List of raw edge weights
    scale : float
        Scale factor for the normalized weights
    base : float
        Base value to add to normalized weights
        
    Returns:
    --------
    list
        Normalized weights with improved scaling
    """
    if not weights:
        return None
        
    min_w, max_w = min(weights), max(weights)
    
    # Handle the case where all weights are the same
    if max_w - min_w == 0:
        return [scale] * len(weights)
    
    # Use power law scaling to emphasize differences between weights
    # This prevents edge weights from being too similar
    
    # First normalize to range [0,1]
    normalized = [(w - min_w) / (max_w - min_w) for w in weights]
    
    # Apply power scaling (amplifies differences)
    power_scaled = [pow(n, 0.5) for n in normalized]  # sqrt is less aggressive than square
    
    # Scale to desired range
    return [n * scale + base for n in power_scaled]

# Computes the Jaccard similarity: size of intersection / union.
def compute_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 0 if union == 0 else intersection / union

def build_context_vectors(sentences, selected_words, all_candidates, context_window):
    """
    Build co-occurrence vectors for each selected word based on surrounding context within a sliding window.
    """
    vectors = {}
    for word in tqdm(selected_words, desc="Building context vectors"): # Add time check 
        vec = np.zeros(len(all_candidates))
        for sent in sentences:
            for i, w in enumerate(sent):
                if w == word:
                    window = sent[max(0, i - context_window):i + context_window + 1]
                    for context_word in window:
                        if context_word in all_candidates:
                            idx = all_candidates.index(context_word)
                            vec[idx] += 1
        vectors[word] = vec
    return vectors

def build_pmi_vectors(sentences, selected_words, all_candidates, context_window):
    """
    Build Positive Pointwise Mutual Information (PMI) vectors for each selected word based on co-occurrence within a sliding context window.
    """
    cooc_matrix = np.zeros((len(selected_words), len(all_candidates)))
    word_counts = {w: 0 for w in selected_words}
    context_counts = {w: 0 for w in all_candidates}
    total_windows = 0

    for sent in tqdm(sentences, desc="Building PMI co-occurrences"): # Add time check 
        for i, word in enumerate(sent):
            if word in selected_words:
                word_counts[word] += 1
                window = sent[max(0, i - context_window):i + context_window + 1]
                for context_word in window:
                    if context_word in all_candidates:
                        idx1 = selected_words.index(word)
                        idx2 = all_candidates.index(context_word)
                        cooc_matrix[idx1][idx2] += 1
                        context_counts[context_word] += 1
                        total_windows += 1

    epsilon = 1e-10
    pmi_matrix = np.zeros_like(cooc_matrix)
    for i, word in enumerate(selected_words):
        for j, context_word in enumerate(all_candidates):
            if cooc_matrix[i][j] > 0:
                p_ij = max(cooc_matrix[i][j], epsilon) / total_windows
                p_i = max(word_counts[word], epsilon) / total_windows
                p_j = max(context_counts[context_word], epsilon) / total_windows
                pmi = np.log2(p_ij / (p_i * p_j))
                if np.isfinite(pmi):
                    pmi_matrix[i][j] = max(pmi, 0)
    return {selected_words[i]: pmi_matrix[i] for i in range(len(selected_words))}

def build_tfidf_vectors(sentences, selected_words, all_candidates, context_window):
    """
    Build TF-IDF-weighted context vectors for selected words by averaging scores 
    of surrounding words within a sliding window across all input sentences.
    """
    docs = [" ".join(sent) for sent in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(tfidf_feature_names, tfidf_matrix.mean(axis=0).A1))

    vectors = {}
    for word in tqdm(selected_words, desc="Building TF-IDF vectors"): # Add time check 
        vec = np.zeros(len(all_candidates))
        for sent in sentences:
            for i, w in enumerate(sent):
                if w == word:
                    window = sent[max(0, i - context_window):i + context_window + 1]
                    for context_word in window:
                        if context_word in all_candidates:
                            idx = all_candidates.index(context_word)
                            vec[idx] += tfidf_scores.get(context_word, 1.0)
        vectors[word] = vec
    return vectors

def compute_cosine_similarity_matrix(word_vectors, word_list):
    """
    Compute a pairwise cosine similarity matrix between word vectors for a given list of words.
    """

    similarity_matrix = np.zeros((len(word_list), len(word_list)))
    for i, w1 in enumerate(word_list):
        v1 = word_vectors.get(w1)
        for j, w2 in enumerate(word_list):
            v2 = word_vectors.get(w2)
            if v1 is not None and v2 is not None and np.sum(v1) > 0 and np.sum(v2) > 0:
                similarity_matrix[i][j] = cosine_similarity([v1], [v2])[0][0]
    return similarity_matrix

def run_jaccard(sentences, selected_words, context_window, all_candidates, cache_file): # Get Jaccard similarity matrix 
    """ 
    Compute the Jaccard similarity matrix for selected words based on their surrounding context windows.
    """
    jaccard_matrix = np.zeros((len(selected_words), len(selected_words)))
    for i, word1 in enumerate(selected_words):
        for j, word2 in enumerate(selected_words):
            if word1 != word2:
                contexts_word1 = [set(sent[max(0, idx - context_window):idx + context_window + 1]) 
                                  for sent in sentences 
                                  for idx, w in enumerate(sent) if w == word1]
                contexts_word2 = [set(sent[max(0, idx - context_window):idx + context_window + 1]) 
                                  for sent in sentences 
                                  for idx, w in enumerate(sent) if w == word2]
                word1_context = set(itertools.chain.from_iterable(contexts_word1))
                word2_context = set(itertools.chain.from_iterable(contexts_word2))
                intersection = len(word1_context & word2_context)
                union = len(word1_context | word2_context)
                jaccard_matrix[i][j] = intersection / union if union > 0 else 0.0
    filtered_embeddings = {selected_words[i]: jaccard_matrix[i] for i in range(len(selected_words))}
    with open(cache_file, 'wb') as f:
        pickle.dump((filtered_embeddings, None, jaccard_matrix), f)
    return filtered_embeddings, None, jaccard_matrix


def run_pmi(sentences, selected_words, context_window, cache_file, epsilon=1e-9):
    """
    Compute a Positive PMI (PPMI) square matrix: seed_words × seed_words
    """
    cooc_matrix = np.zeros((len(selected_words), len(selected_words)))
    word_counts = Counter()
    total_windows = 0

    for sent in sentences:
        for i, word1 in enumerate(sent):
            if word1 not in selected_words:
                continue
            word_counts[word1] += 1
            window = sent[max(0, i - context_window): i + context_window + 1]
            for word2 in window:
                if word2 in selected_words and word2 != word1:
                    idx1, idx2 = selected_words.index(word1), selected_words.index(word2)
                    cooc_matrix[idx1, idx2] += 1
                    total_windows += 1

    if total_windows == 0:
        warnings.warn("No co-occurrences found. Returning empty results.")
        return {}, None, np.zeros((len(selected_words), len(selected_words)))

    # ---- compute PPMI ----
    pmi_matrix = np.zeros_like(cooc_matrix, dtype=float)
    for i, w1 in enumerate(selected_words):
        for j, w2 in enumerate(selected_words):
            if cooc_matrix[i, j] > 0:
                p_ij = cooc_matrix[i, j] / total_windows
                p_i = word_counts[w1] / total_windows
                p_j = word_counts[w2] / total_windows
                if p_i > 0 and p_j > 0:
                    pmi = np.log2(p_ij / (p_i * p_j + epsilon))
                    pmi_matrix[i, j] = max(0, pmi)  # PPMI

    filtered_embeddings = {selected_words[i]: pmi_matrix[i] for i in range(len(selected_words))}
    with open(cache_file, "wb") as f:
        pickle.dump((filtered_embeddings, None, pmi_matrix), f)

    return filtered_embeddings, None, pmi_matrix

def run_tfidf_overlap(sentences, selected_words, context_window, all_candidates, cache_file): # Get TF-IDF co-occurrnece matrix 
    """
    Calculates similarity based on TF-IDF vectors of context windows.
    The primary and recommended method is 'cosine' similarity.
    The 'experimental_unnormalized_sum' is retained for legacy purposes but is not recommended.
    """

    docs = [" ".join(sent) for sent in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix_raw = vectorizer.fit_transform(docs)
    tfidf_feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(tfidf_feature_names, tfidf_matrix_raw.mean(axis=0).A1))
    tfidf_matrix = np.zeros((len(selected_words), len(selected_words)))
    for i, word1 in enumerate(selected_words):
        for j, word2 in enumerate(selected_words):
            if word1 != word2:
                contexts_word1 = [set(sent[max(0, idx - context_window):idx + context_window + 1]) 
                                  for sent in sentences 
                                  for idx, w in enumerate(sent) if w == word1]
                contexts_word2 = [set(sent[max(0, idx - context_window):idx + context_window + 1]) 
                                  for sent in sentences 
                                  for idx, w in enumerate(sent) if w == word2]
                word1_context = set(itertools.chain.from_iterable(contexts_word1))
                word2_context = set(itertools.chain.from_iterable(contexts_word2))
                shared = word1_context & word2_context
                score = sum([tfidf_scores.get(tok, 1.0) for tok in shared])
                tfidf_matrix[i][j] = score
    filtered_embeddings = {selected_words[i]: tfidf_matrix[i] for i in range(len(selected_words))}
    with open(cache_file, 'wb') as f:
        pickle.dump((filtered_embeddings, None, tfidf_matrix), f)
    return filtered_embeddings, None, tfidf_matrix

# Remove possessives, quotes, and hyphens
def clean_words(words):
    """Clean word list by removing possessives, quotes, and hyphens"""
    cleaned = []
    for word in words:
        word = word.replace("'", "'")        
        word = re.sub(r"'s$|s'$", "", word)          
        word = word.replace("-", "")           
        word = word.strip()                        
        if word:                                      
            cleaned.append(word)
    return cleaned

# Replaces synonyms in a text with their group label (e.g., "dementia", "alzheimers" → "Dementia").
def replace_group_words(text, seed_groups):
    """Replace synonym words with their group label"""
    for group_label, words in seed_groups.items():
        pattern = r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\b'
        text = re.sub(pattern, group_label, text, flags=re.IGNORECASE)
    return text

# Split long text into chunks for token limitation
def split_into_chunks(text, max_tokens=512): 
    """Split long text into chunks under max_tokens using sentence boundaries"""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    semantic_categories=None,
    for sentence in sentences:
        sent_tokens = len(word_tokenize(sentence))
        if sent_tokens > max_tokens:
            # If the sentence is too long, split on punctuation
            subs = re.split(r'[.!?]', sentence)
            for sub in subs:
                sub = sub.strip()
                if not sub:
                    continue
                sub_len = len(word_tokenize(sub))
                if current_tokens + sub_len <= max_tokens:
                    current_chunk.append(sub)
                    current_tokens += sub_len
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sub]
                    current_tokens = sub_len
        else:
            if current_tokens + sent_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sent_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Lets users interactively assign colors and labels to groups of words.
def get_custom_groups(words):

    """Let user define custom word groups with colors and labels"""

    print("\n--- Customize Semantic Network Groups ---")
    print("Define word groups with a color and label. Enter 'done' when finished.")
    print("Example:")
    print("  Label: Memory Terms")
    print("  Color: blue")
    print("  Words: dementia, recall, forget")
    print(f"\nAvailable words (first 30): {', '.join(words[:30])}... (total: {len(words)})\n")

    groups = {}
    while True:
        label = input("Enter group label (or 'done' to finish): ").strip()
        if not label:
            print("Label cannot be empty. Please enter a label or type 'done' to exit.")
            continue
        if label.lower() in ['done', 'exit', 'quit']:
            print("Exiting group input.")
            break
        color = input(f"Pick a color for group '{label}' (e.g., blue, green, orange): ").strip()
        group_words = input(f"Enter words for group '{label}' (comma-separated): ").strip().split(",")
        # Normalize to lowercase
        group_words = [w.strip().lower() for w in group_words if w.strip()]
        groups[label] = {'color': color, 'words': group_words}
        print(f"✔️ Added group '{label}' with color '{color}' and words: {group_words}\n")
    return groups

#---------------------------------------------------------------------------------
# TEXT PROCESSING FUNCTIONS
#---------------------------------------------------------------------------------

# Takes a list of sentences, tokenizes them, removes stop words and punctuation,
# and lemmatizes words (e.g., "running" → "run").
def tokenize_and_filter(sentences, stop_list, lemmatize=True, cross_pos_normalize=False):
    """Tokenize and filter text, optionally applying lemmatization"""
    # Build reverse mapping
    word_to_base = {}
    for base_word, variants in WORD_FAMILIES.items():
        for variant in variants:
            word_to_base[variant.lower()] = base_word
    
    # Helper function to normalize a single word using word families
    def normalize_with_families(word):
        word_lower = word.lower()
        if word_lower in word_to_base:
            return word_to_base[word_lower]
        return word_lower
    
    words = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
            
        tokens = word_tokenize(sentence)
        tagged_tokens = pos_tag(tokens)
        
        filtered_tokens = []
        for token, tag in tagged_tokens:
            if token.lower() not in stop_list and token.isalpha() and re.match(r'^[a-zA-Z-]+$', token) and token.lower() not in stop_list:
                if lemmatize:
                    # Try multiple POS forms and take the shortest
                    token_lower = token.lower()
                    lemma_noun = lemmatizer.lemmatize(token_lower, pos='n')
                    lemma_verb = lemmatizer.lemmatize(token_lower, pos='v')
                    lemma_adj = lemmatizer.lemmatize(token_lower, pos='a')
                    token = min([lemma_noun, lemma_verb, lemma_adj], key=len)
                    
                    # Apply word family normalization
                    token_lower = token.lower()
                    if token_lower in word_to_base:
                        token = word_to_base[token_lower]
                else:
                    token = token.lower()
                    # Apply word family even without lemmatization
                    if token in word_to_base:
                        token = word_to_base[token]
                filtered_tokens.append(token)
        
        if filtered_tokens:  # Only add non-empty lists
            words.extend(filtered_tokens)
    
    # Apply cross-POS normalization if requested
    if cross_pos_normalize and words:
        # Use stemming to find common roots
        stemmer = SnowballStemmer('english')
        word_stems = {word: stemmer.stem(word) for word in words}
        
        # Group words by stems
        stem_to_words = {}
        for word, stem in word_stems.items():
            stem_to_words.setdefault(stem, []).append(word)
        
        # Replace words with their normalized forms
        normalized_words = []
        already_included = set()
        for word in words:
            stem = word_stems[word]
            if stem not in already_included:
                normalized_words.append(min(stem_to_words[stem], key=len))
                already_included.add(stem)
            
        return normalized_words
    
    return words

# Like tokenize_and_filter but works on a list of words (not full sentences).
# Cleans and lemmatizes individual words.
def normalize_words(words, stop_list, lemmatize=True, cross_pos_normalize=False):

    """Normalize a list of words with lemmatization and/or cross-POS normalization"""

    normalized = []
    tagged_tokens = pos_tag(words)
    
    # Build a reverse mapping for easier lookup
    word_to_base = {}
    for base_word, variants in WORD_FAMILIES.items():
        for variant in variants:
            word_to_base[variant] = base_word
    
    # First pass: standard lemmatization
    lemmatized_words = []
    for token, tag in tagged_tokens:
        if token.lower() not in stop_list and token.isalpha() and re.match(r'^[a-zA-Z-]+$', token):
            if lemmatize:
                # Try multiple POS to get the shortest lemma
                token_lower = token.lower()
                lemma_noun = lemmatizer.lemmatize(token_lower, pos='n')
                lemma_verb = lemmatizer.lemmatize(token_lower, pos='v')
                lemma_adj = lemmatizer.lemmatize(token_lower, pos='a')
                token = min([lemma_noun, lemma_verb, lemma_adj], key=len)
                
                # Check if the original or lemmatized token is in our word families
                if token_lower in word_to_base:
                    token = word_to_base[token_lower]
                elif token in word_to_base:
                    token = word_to_base[token]
            else:
                token = token.lower()
                # Even if not lemmatizing, still normalize word families
                if token in word_to_base:
                    token = word_to_base[token]
            
            lemmatized_words.append(token)
    
    # Second pass: cross-POS normalization if requested
    if cross_pos_normalize:
        # Use stemming to find common roots
        stemmer = SnowballStemmer('english')
        word_stems = {word: stemmer.stem(word) for word in lemmatized_words}
        
        # Group words by stems
        stem_to_words = {}
        for word, stem in word_stems.items():
            stem_to_words.setdefault(stem, []).append(word)
        
        # Report word families found
        word_families_found = {
            min(words, key=len): words 
            for stem, words in stem_to_words.items() 
            if len(words) > 1
        }
        
        if word_families_found:
            print("Auto-detected word families:")
            for root, family in word_families_found.items():
                print(f"  - {root}: {', '.join(w for w in family if w != root)}")
        
        # Return the shortest word from each family
        normalized = []
        already_included = set()
        for word in lemmatized_words:
            stem = word_stems[word]
            if stem not in already_included:
                normalized.append(min(stem_to_words[stem], key=len))
                already_included.add(stem)
    else:
        normalized = lemmatized_words
    
    return normalized

# Given some seed words and a word embedding dictionary, it finds other words
# that are similar based on cosine similarity.
def find_semantically_similar_words(seed_words, word_embeddings, threshold=0.7):

    """Find words semantically similar to seed words using cosine similarity"""
    
    similar_words = []
    seed_embeddings = [word_embeddings[word] for word in seed_words if word in word_embeddings]

    for word, embedding in word_embeddings.items():
        if word not in seed_words:
            similarity_scores = [cosine_similarity([embedding], [seed_embedding])[0][0] for seed_embedding in seed_embeddings]
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            if avg_similarity >= threshold:
                similar_words.append((word, avg_similarity))

    similar_words = sorted(similar_words, key=lambda x: x[1], reverse=True)
    return similar_words


# Loaded successfully message

# ----------------------------------------------------------
# Embedding and Clustering 
#---------------------------------------------------------------------------------
# EMBEDDING AND CLUSTERING FUNCTIONS
#---------------------------------------------------------------------------------

# Core engine to generate the word embeddings and similarity/co-occurrence matrices.
def train_embedding(sentences, context_window, stop_list, seed_words, clustering_method, 
                    num_words, lemmatize=True, min_word_frequency=2, reuse_clusterings=True, 
                    cross_pos_normalize=False, distance_metric="default", custom_word_filter=None, 
                    semantic_categories=None, 
                    link_threshold=None, link_color_threshold=None):
    """
    Generate word embeddings and similarity/co-occurrence matrices based on the specified clustering method.
    
    Parameters:
    -----------
    sentences : list
        List of text sentences to process
    context_window : int
        Window size for co-occurrence calculation
    stop_list : set
        Set of stop words to exclude
    seed_words : list
        List of seed words to base similarity on
    clustering_method : int
        1=RoBERTa embeddings, 2=Jaccard/Cosine, 3=PMI, 4=TF-IDF
    num_words : int
        Number of words to include in the network
    lemmatize : bool
        Whether to apply lemmatization
    min_word_frequency : int
        Minimum frequency for a word to be included
    reuse_clusterings : bool
        Whether to reuse cached embeddings
    cross_pos_normalize : bool
        Whether to normalize across parts of speech
    distance_metric : str
        "default" = cosine similarity (same as "cosine")
        "cosine" = cosine similarity between context vectors  
        "raw_weighted" = raw weighted co-occurrence scores
    custom_word_filter : function
        Optional function to filter words
        
    Returns:
    --------
    tuple
        (word_embeddings, similarity_matrix, co_occurrence_matrix)
    """

    
    print(f"Number of words analyzed is {num_words}")
    print(f"Using distance metric: {distance_metric}")
    
    # Build reverse mapping for word families
    word_to_base = {}
    
    for base_word, variants in WORD_FAMILIES.items():
        for variant in variants:
            word_to_base[variant.lower()] = base_word
    
    # Create a unique cache name based on parameters
    cache_id = f"{clustering_method}_{num_words}_{context_window}_{min_word_frequency}_{lemmatize}_{distance_metric}"
    corpus_hash = hashlib.md5(str(sentences[:200]).encode()).hexdigest()[:10]
    cache_file = os.path.join(CLUSTERING_DIR, f"embeddings_{cache_id}_{corpus_hash}.pkl")
    
    print("Cache path EXPECTED: ", cache_file)
    print("File exists? --> ", os.path.exists(cache_file))
    # Check for cached embeddings
    if reuse_clusterings and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Verify cache format based on clustering method
            if clustering_method == 1 and len(cached_data) == 3 and cached_data[1] is not None:
                return cached_data
            elif clustering_method in [2, 3, 4] and len(cached_data) == 3 and cached_data[2] is not None:
                return cached_data
            else:
                print("Cache format doesn't match requested clustering method. Regenerating clusterings.")
        except Exception as e:
            print(f"Error loading cached clusterings: {e}. Regenerating clusterings.")

    # Tokenize and filter sentences
    print("Tokenizing sentences...")
    tokenized_sentences = [tokenize_and_filter([sentence], stop_list, lemmatize, cross_pos_normalize) for sentence in sentences]
    
    # Apply word family normalization to all tokens
    normalized_sentences = []
    for sent in tokenized_sentences:
        normalized_sent = []
        for word in sent:
            word_lower = word.lower()
            if word_lower in word_to_base:
                normalized_sent.append(word_to_base[word_lower])
            else:
                normalized_sent.append(word)
        normalized_sentences.append(normalized_sent)
    
    processed_sentences = [[word for word in sent] for sent in normalized_sentences]
    flat_text = " ".join([" ".join(sent) for sent in processed_sentences])
    all_words = [word for sent in processed_sentences for word in sent]
    print(f"Tokenization complete. Processed {len(processed_sentences)} sentences.")
    print(f"Total words: {len(all_words)} | Unique words: {len(set(all_words))}")
    print("Counting word frequencies...")
    word_frequencies = Counter(all_words)
    print(f"Top {num_words} most frequent words: {word_frequencies.most_common(num_words)}")

    # Apply word family normalization to frequency counts
    normalized_frequencies = Counter()
    for word, count in word_frequencies.items():
        word_lower = word.lower()
        if word_lower in word_to_base:
            normalized_word = word_to_base[word_lower]
            normalized_frequencies[normalized_word] += count
        else:
            normalized_frequencies[word] += count

    word_frequencies = normalized_frequencies
    print(f"After word family normalization - Top {num_words} most frequent words: {word_frequencies.most_common(num_words)}")

    # Apply custom filter to candidates if provided
    if custom_word_filter:
        all_candidates = [word for word, _ in word_frequencies.items() 
                         if word_frequencies[word] >= min_word_frequency and custom_word_filter(word)]
    else:
        all_candidates = [word for word, _ in word_frequencies.items() 
                         if word_frequencies[word] >= min_word_frequency]
                         
    print("Finding seed words...")

    # Normalize seed words using the same word families
    normalized_seed_words = []
    for seed in seed_words:
        seed_lower = seed.lower()
        if seed_lower in word_to_base:
            normalized_seed_words.append(word_to_base[seed_lower])
        else:
            normalized_seed_words.append(seed)
    
    # Remove duplicates
    normalized_seed_words = list(set(normalized_seed_words))
    
    missing_seeds = [w for w in normalized_seed_words if w not in all_candidates]
    if missing_seeds:
        print(f"⚠️ These seed words were not included in embeddings due to low frequency: {missing_seeds}")
    
    seed_words = [word for word in normalized_seed_words if word in all_candidates]

    if not seed_words: # Check valid length of seed words
        print(f"No seed words found in the corpus under min_word_frequency: {min_word_frequency}. Please check your input.")
        return None, None, None
    
    # Get top co-occurring words with seed_words
    cooccur_counts = Counter()
    for sent in processed_sentences:
        for i, word in enumerate(sent):
            if word in seed_words:
                window = sent[max(0, i - context_window):i + context_window + 1]
                for w in window:
                    if w != word and w not in seed_words and word_frequencies[w] >= min_word_frequency:
                        cooccur_counts[w] += 1
    top_related = [w for w, _ in cooccur_counts.most_common(num_words - len(seed_words))]
    selected_words = seed_words + top_related

    print("Top co-occurring words with seed:")
    print(cooccur_counts.most_common(10))

    if clustering_method == 1:  # RoBERTa token-based similarity
        print("Generating RoBERTa token-based embeddings...")
        #######################
        # Load model and tokenizer
        tokenizer = TOKENIZER
        model = MODEL
        device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        model.to(device)
        model.eval()
        word_embeddings = {}
        
        print("Extracting word embeddings across paragraphs...")
        for paragraph in tqdm(sentences):
            chunks = split_into_chunks(paragraph, max_tokens=MAX_TOKENS)
            for chunk in chunks: 
                inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True).to(device)
                offset_mapping = inputs.pop("offset_mapping")
                with torch.no_grad():
                    outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                current_word = ""
                current_vecs = []

            for i, token in enumerate(tokens):
                if token in ["[CLS]", "[SEP]", "<s>", "</s>"] or not token.isalpha():
                    continue
                if token.startswith("Ġ"):
                    if current_word and current_vecs:
                        lemmatized = lemmatizer.lemmatize(current_word, pos='n')
                        word_embeddings.setdefault(lemmatized, []).append(np.mean(current_vecs, axis=0))
                    current_word = token[1:].lower()
                    current_vecs = [hidden_states[i]]
                else:
                    current_word += token.lower()
                    current_vecs.append(hidden_states[i])
            if current_word and current_vecs:
                lemmatized = lemmatizer.lemmatize(current_word, pos='n')
                word_embeddings.setdefault(lemmatized, []).append(np.mean(current_vecs, axis=0))
                
        print(f"Intermediate: {len(word_embeddings)} lemmatized words extracted (may include stop words).")
        print("Sample captured words (pre-filter):", list(word_embeddings.keys())[:10])

        averaged_embeddings = {
            word: np.mean(vecs, axis=0)
            for word, vecs in word_embeddings.items()
            if len(vecs) >= 2
        }

        candidate_words = [w for w in all_candidates if w not in seed_words and w in averaged_embeddings]
        seed_embeddings = [averaged_embeddings[word] for word in seed_words if word in averaged_embeddings]

        word_scores = []
        for word in candidate_words:
            embedding = averaged_embeddings[word]
            scores = [cosine_similarity([embedding], [seed_vec])[0][0] for seed_vec in seed_embeddings]
            avg_score = sum(scores) / len(scores) if scores else 0
            word_scores.append((word, avg_score))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        top_similar_words = [w for w, _ in word_scores[:num_words - len(seed_words)]]
        selected_words = seed_words + top_similar_words
        print("Selected seed + top similar words:", selected_words)

        filtered_embeddings = {word: averaged_embeddings[word] for word in selected_words if word in averaged_embeddings}
        embedding_vectors = np.array(list(filtered_embeddings.values()))
        similarity_matrix = cosine_similarity(embedding_vectors)
        # Save embeddings to cache
        with open(cache_file, 'wb') as f:
            pickle.dump((filtered_embeddings, similarity_matrix, None), f)

        return filtered_embeddings, similarity_matrix, None

    elif clustering_method == 2:  # Jaccard 
        if distance_metric == "cosine":
            print(f"Computing cosine similarity for co-occurrence context window using {distance_metric} metric...")
            vectors = build_context_vectors(processed_sentences, selected_words, all_candidates, context_window)
            similarity_matrix = compute_cosine_similarity_matrix(vectors, selected_words)
            filtered_embeddings = {w: similarity_matrix[i] for i, w in enumerate(selected_words)}
            with open(cache_file, 'wb') as f:
                pickle.dump((filtered_embeddings, similarity_matrix, None), f)
            return filtered_embeddings, similarity_matrix, None
        else:
            print(f"Computing context-window overlap by Jaccard using {distance_metric} metric...")
            return run_jaccard(processed_sentences, selected_words, context_window, all_candidates, cache_file)


    elif clustering_method == 3:  # PMI
        if distance_metric == 'cosine':
            print(f"Computing cosine similarity for PMI context window using {distance_metric} metric...")
            vectors = build_pmi_vectors(processed_sentences, selected_words, all_candidates, context_window)
            print(f"PMI vectors keys: {list(vectors.keys())}")
            filtered_words = [w for w in selected_words if w in vectors and np.sum(vectors[w]) > 0]
            if not filtered_words:
                print("⚠️ No valid PMI vectors for any selected words.")
                return None, None, None
            similarity_matrix = compute_cosine_similarity_matrix(vectors, filtered_words)
            filtered_embeddings = {w: similarity_matrix[i] for i, w in enumerate(filtered_words)}
            with open(cache_file, 'wb') as f:
                pickle.dump((filtered_embeddings, similarity_matrix, None), f)
            return filtered_embeddings, similarity_matrix, None
    

    elif clustering_method == 4:  # TF-IDF weighted co-occurrence

        if distance_metric in {'cosine', 'default'}:
            print(f"Computing cosine similarity for TF-IDF context window using cosine metric...")
            vectors = build_tfidf_vectors(processed_sentences, selected_words, all_candidates, context_window)
            similarity_matrix = compute_cosine_similarity_matrix(vectors, selected_words)
            # Create word embeddings from TF-IDF weighted cosine similarity matrix
            filtered_embeddings = {w: similarity_matrix[i] for i, w in enumerate(selected_words)}
            
            # Save embeddings to cache
            with open(cache_file, 'wb') as f:
                pickle.dump((filtered_embeddings, similarity_matrix, None), f)
            
            return filtered_embeddings,similarity_matrix, None
        else:
            # Original TF-IDF implementation
            print(f"Computing co-occurrence for TF-IDF context window using {distance_metric} metric...")
            print("WARNING: You are using the 'experimental_unnormalized_sum' metric for TF-IDF. "
              "This is an unnormalized, non-standard method. For scientifically valid results, "
              "it is strongly recommended to use distance_metric='cosine'.")
            return run_tfidf_overlap(processed_sentences, selected_words, context_window, all_candidates, cache_file)

    else:
        print("Invalid clustering method. Choose 1 (RoBERTa), 2 (Jaccard), 3 (PMI), or 4 (TF-IDF).")
        return None, None, None
#---------------------------------------------------------------------------------
# Visualization 
## Heatmap 
#---------------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS: Heatmaps
#---------------------------------------------------------------------------------
# Function to Generate Word Similarity Heatmap

def plot_heatmap(clustering_method, word_embeddings, similarity_matrix, co_occurrence_matrix, distance_metric, clustered=False):
    
    """
    Generate a word relationship heatmap (and clustered version) using similarity or co-occurrence data.

    Parameters
    ----------
    clustering_method    : int     1 = RoBERTa, 2 = Jaccard, 3 = PMI, 4 = TF-IDF
    word_embeddings      : dict    keyed by word → vector (similarity or co-occurrence)
    similarity_matrix    : np.ndarray | None   used if clustering_method = 1 or distance_metric = "cosine"
    co_occurrence_matrix : np.ndarray | None   used if clustering_method in 2–4 and distance_metric = "default"
    distance_metric      : str     "default" = co-occurrence | "cosine" = similarity from vectors
    clustered            : bool    if True, also shows a dendrogram-clustered version of the heatmap

    Returns
    -------
    None. Displays heatmaps and saves clustered version to "clustermap_temp.png".

    Notes
    -----
    - Brighter colors indicate stronger word relationships.
    - Dendrograms help reveal natural groupings of words.
    - This visualization is useful for understanding semantic or thematic clusters in your data.
    """


    words = list(word_embeddings.keys())
    METHOD_NAMES = {
    1: "RoBERTa",
    2: "Jaccard",
    3: "PMI",
    4: "TF-IDF"}

    if clustering_method == 1 and similarity_matrix is not None:
        matrix = similarity_matrix
        title = "RoBERTa Word Similarity"
    elif clustering_method == 2:  # Jaccard
        if distance_metric == "cosine" and similarity_matrix is not None:
            matrix = similarity_matrix
            title = "Jaccard (Context Cosine Similarity)"
        elif distance_metric == "default" and co_occurrence_matrix is not None:
            matrix = co_occurrence_matrix
            title = "Jaccard (Co-Occurrence)"
        else:
            print("Error: No valid matrix available for Jaccard.")
            return

    elif clustering_method == 3:  # PMI
        if distance_metric == "cosine" and similarity_matrix is not None:
            matrix = similarity_matrix
            title = "PMI (Context Cosine Similarity)"
        else:
            print("Error: PMI currently only supports cosine similarity.")
            return
            
    elif clustering_method == 4:
        if distance_metric in {'cosine', 'default'} and similarity_matrix is not None:
            matrix = similarity_matrix 
            title = "Context Cosine Similarity" 
        elif distance_metric == 'raw_weighted' and co_occurrence_matrix is not None:
            matrix = co_occurrence_matrix 
            title = "Co-Occurrence (Experimental Unnormalized Sum)"
        else:
            print("Error: No valid matrix available.")
            return
    else:
        print("Error: Invalid matrix input or clustering method.")
        return

    matched_df = pd.DataFrame(matrix, index=words, columns=words)

    cluster = sns.clustermap(matched_df,
                         annot=True,
                         fmt='.2f',
                         cmap='inferno',
                         row_linkage=hierarchy.linkage(matrix, method='ward'),
                         col_linkage=hierarchy.linkage(matrix.T, method='ward'),
                         figsize=(10, 8),
                         annot_kws={"size": 8},
                         dendrogram_ratio=(0.1, 0.1),
                         cbar_pos=(0.02, 0.8, 0.03, 0.18))

    cluster.savefig("clustermap_temp.png")  
    plt.close(cluster.fig)  

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 

    # Non-clustered Heatmap 
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(matched_df, annot=True, fmt='.2f', cmap='inferno', ax=ax1,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.5})
    ax1.set_title("Heatmap (unclustered)")

    # Clustered Heatmap 
    ax2 = fig.add_subplot(gs[1])
    img = mpimg.imread("clustermap_temp.png")
    ax2.imshow(img)
    ax2.axis('off')
    ax2.set_title("Heatmap (clustered)")

    plt.tight_layout()
    
    print(f"\nHeatmap visualization saved to: clustermap_temp.png")
    print("\nAbout Heatmap visualization:")
    print("Heatmaps help visualize relationships between words in your dataset. In these plots:")
    print("- Brighter colors indicate stronger relationships between word pairs")
    print("- The non-clustered heatmap shows the raw relationship matrix")
    print("- The clustered heatmap groups similar words together using hierarchical clustering")
    print("- Dendrograms on the sides show the hierarchical structure of word relationships")
    print("- This visualization helps identify patterns and word clusters in your text data")
    
    print(f"\nVisualization type: {title}")
    print("1. Word Similarity: Shows semantic similarity between word vectors")
    print("2. Co-Occurrence: Shows how frequently words appear together")
    print("3. PMI: Highlights statistically significant word associations")
    print("4. TF-IDF: Emphasizes important terms while downweighting common words")

    return fig

## tSNE

#---------------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS: t-SNE 
#---------------------------------------------------------------------------------

def plot_tsne_dimensional_reduction(
        word_embeddings: dict,
        *,
        similarity_matrix: np.ndarray | None = None,
        co_occurrence_matrix: np.ndarray | None = None,
        clustering_method: int = 1,
        seed_words: list[str] | None = None,
        distance_metric
    ) -> None:
    """
    2-D t-SNE visualisation of the local semantic space.

    Parameters
    ----------
    word_embeddings: dict   keyed by word → embedding vector
    similarity_matrix: np.ndarray | None (used for RoBERTa = 1)
    co_occurrence_matrix: np.ndarray | None   (used for 2-4)
    clustering_method: int     1 = RoBERTa, 2 = Jaccard, 3 = PMI, 4 = TF-IDF
    seed_words: list[str] | None
                           – words to highlight. If None a small default list
                             is used so the plot still renders.
    distance_metric: str     "default" = co-occurrence | "cosine" = similarity from vectors | "raw_weighted = raw weighted co-occurrence scores
    """

    # Choose clustering/embedding method
    if clustering_method == 1 and similarity_matrix is not None:
        matrix = similarity_matrix
    elif clustering_method == 2:  # Jaccard
        if distance_metric == "cosine" and similarity_matrix is not None:
            matrix = similarity_matrix
        elif distance_metric == "default" and co_occurrence_matrix is not None:
            matrix = co_occurrence_matrix
        else:
            print("Error: missing matrix for Jaccard t-SNE visualisation.")
            return

    elif clustering_method == 3:  # PMI
        if distance_metric == "cosine" and similarity_matrix is not None:
            matrix = similarity_matrix
        else:
            print("Error: PMI currently only supports cosine similarity for t-SNE.")
            return

    elif clustering_method == 4:
        if distance_metric in {'cosine','default'} and similarity_matrix is not None:
            matrix = similarity_matrix
        elif distance_metric == "raw_weighted" and co_occurrence_matrix is not None:
            matrix = co_occurrence_matrix
        else:
            print("Error: missing matrix for t-SNE visualisation.")
            return
    else:
        print("Error: unsupported clustering method for t-SNE.")
        return

    # Prep matrix 
    words = list(word_embeddings.keys())
    df_matrix = pd.DataFrame(matrix, index=words, columns=words)
    print(f"df matrix index is", df_matrix.index)
    if seed_words is None or len(seed_words) == 0:
        # tiny default so the function never crashes
        seed_words = ["education", "learning", "teaching", "student", "school", "classroom", "curriculum", "academic"]

    print(f"seed_words is", seed_words)
    # keep only seeds present in the matrix
    present_seeds = [w for w in seed_words if w in df_matrix.index]
    if not present_seeds:
        print("Warning: none of the supplied seed words appear in the vocabulary; "
              "plotting without highlighted seeds.")
    else:
        missing = set(seed_words) - set(present_seeds)
        if missing:
            print(f"Warning: the following seed words were not found and will be skipped: {missing}")

    # collect candidate item around the seed words
    neighbours: set[str] = set(present_seeds)
    for s in present_seeds:
        # choose a modest threshold; it is *not* the plotting threshold
        threshold = 0.3 if clustering_method == 1 else 0.1
        nearby = df_matrix[s][df_matrix[s] > threshold].index.tolist()
        neighbours.update(nearby)

    # basic filtering of stop-words & very common fillers
    stop_words_std = set(stopwords.words("english"))
    fillers       = {"the", "and", "to", "of", "a", "in", "is", "it", "that",
                     "was", "for", "on", "with", "as", "be", "at", "by",
                     "have", "are", "this"}
    keep = [w for w in neighbours if w not in stop_words_std | fillers]

    # guarantee the seeds are kept
    for s in present_seeds:
        if s not in keep:
            keep.append(s)

    if len(keep) < 3:
        print("Not enough words to build a meaningful t-SNE plot.")
        return

    # build 2-D embedding via t-SNE
    sub = df_matrix.loc[keep, keep]

    perplexity = max(2, min(30, len(keep)//3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    emb2d = tsne.fit_transform(sub)

    plot_df = pd.DataFrame({
        "x": emb2d[:, 0],
        "y": emb2d[:, 1],
        "word": keep,
        "is_seed": [w in present_seeds for w in keep]
    })
    
    plt.style.use("default") 
    plt.figure(figsize=(12, 10))

    # non-seeds
    ns = plot_df[~plot_df.is_seed]
    plt.scatter(ns.x, ns.y, s=110, c="#9ecae1", alpha=0.75, edgecolors="none")

    # seeds – bigger & distinct colour
    sd = plot_df[plot_df.is_seed]
    plt.scatter(sd.x, sd.y, s=220, c="#d62728", alpha=0.9, edgecolors="black", linewidths=0.5)

    for _, r in plot_df.iterrows():
        plt.text(r.x, r.y, r.word,
                 fontsize=14 if r.is_seed else 10,
                 fontweight="bold" if r.is_seed else "normal",
                 ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.25",
                           fc="white", ec="black", alpha=0.75 if r.is_seed else 0.55))

    plt.title(f"t-SNE semantic map – highlighted: {', '.join(present_seeds) or 'none'}",
              fontsize=16, fontweight="bold")
    plt.xticks([]); plt.yticks([])
    plt.grid(alpha=0.25)

    # auto-create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "tsne_seed_words.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"t-SNE plot saved → {out_path}")

    ## Semantic network 
    #---------------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS: Semantic Network
#---------------------------------------------------------------------------------
def plot_semantic_network(word_embeddings, seed_words, clustering_method,
                          similarity_matrix=None, co_occurrence_matrix=None,
                          semantic_categories=None,
                          link_threshold=None, link_color_threshold=None,
                          distance_metric="default",
                          network_layout="kamada-kawai"):
    """
    Generate a semantic network graph using word embeddings and contextual relationships.

    Parameters
    ----------
    word_embeddings      : dict           keyed by word → embedding vector or co-occurrence row
    seed_words           : list[str]      words to highlight as central; required
    clustering_method    : int            1 = RoBERTa, 2 = Jaccard, 3 = PMI, 4 = TF-IDF
    similarity_matrix    : np.ndarray     optional; used when method = 1 or distance_metric = "cosine"
    co_occurrence_matrix : np.ndarray     optional; used when method ∈ {2,3,4} and distance_metric = "default"
    semantic_categories  : dict | None    optional; word groupings with assigned colors
    link_threshold       : float | None   optional; minimum edge weight to include in plot
    link_color_threshold : float | None   optional; minimum edge weight to highlight as strong
    distance_metric      : str            "default" = co-occurrence | "cosine" = similarity between vectors
    network_layout       : str            Layout algorithm for positioning nodes.
                                          Options: "spring", "kamada-kawai", "circular", "shell"
                                          Default = "kamada-kawai"

    Returns
    -------
    None. Displays an interactive network plot with labeled nodes and weighted, styled edges.
    """

    G = nx_lib.Graph()
    words = list(word_embeddings.keys())

    default_color = "#1f77b4"
    seed_color = "#FFD700"

    node_colors, node_labels, node_color_dict, node_community_dict = [], {}, {}, {}
    used_categories = set()
    # ------------------------------------------------------------------ #
    # Node colours & community labels                                    #
    # ------------------------------------------------------------------ #
    seed_words_lower = set(w.lower() for w in seed_words)
    for word in words:
        assigned, w_lc = False, word.lower()
        if semantic_categories:
            for label, group in semantic_categories.items():
                group_words_lower = set([w.lower() for w in group['words']])
                if w_lc in group_words_lower:
                    node_colors.append(group['color'])
                    node_color_dict[word] = group['color']
                    node_labels[word] = word
                    node_community_dict[word] = label
                    used_categories.add(label)
                    assigned = True
                    break
        if not assigned:
            if w_lc in seed_words_lower:
                node_colors.append(seed_color)
                node_color_dict[w_lc] = seed_color
                node_labels[w_lc] = word
                node_community_dict[w_lc] = "Seed"
                used_categories.add("Seed Word")
            else:
                node_colors.append(default_color)
                node_color_dict[w_lc] = default_color
                node_labels[w_lc] = word
                node_community_dict[w_lc] = "Regular"
                used_categories.add("Regular")
                
    print(f"seed words in lower",seed_words_lower)
    for word in words:
        G.add_node(word, size=400 if word in seed_words else 100,
                   community=node_community_dict[word])

    # edge construction (unchanged)   
    # Process edges with the same code as before

    if clustering_method == 1 and similarity_matrix is not None:  # RoBERTa similarity 
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j:
                    similarity = similarity_matrix[i][j]
                    if similarity > 0.2:
                        G.add_edge(word1, word2, weight=similarity)
        raw_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_weights = normalize_edge_weights(raw_weights, scale=7.0, base=0.5)

    elif clustering_method == 2:
        if distance_metric == "cosine" and similarity_matrix is not None: # Jaccard cosine
            for i, word1 in enumerate(words):
                for j, word2 in enumerate(words):
                    if i != j:
                        similarity = similarity_matrix[i][j]
                        if similarity > 0.2:
                            G.add_edge(word1, word2, weight=similarity)
            raw_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_weights = normalize_edge_weights(raw_weights, scale=7.0, base=0.5)
            
        elif distance_metric == "default" and co_occurrence_matrix is not None:  # Jaccard co-occurrence
            for i, word1 in enumerate(words):
                for j, word2 in enumerate(words):
                    if i != j:
                        jaccard_score = co_occurrence_matrix[i][j]
                        if jaccard_score > 0.1:
                            G.add_edge(word1, word2, weight=jaccard_score)
            raw_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_weights = normalize_edge_weights(raw_weights, scale=7.0, base=0.5)
        else: 
            print("Error: No valid Jaccard matrix available.")
            return

    elif clustering_method == 3:
        if distance_metric == "cosine" and similarity_matrix is not None:# PMI cosine
            for i, word1 in enumerate(words):
                for j, word2 in enumerate(words):
                    if i != j:
                        similarity = similarity_matrix[i][j]
                        if similarity > 0.1:
                            G.add_edge(word1, word2, weight=similarity)
            raw_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_weights = normalize_edge_weights(raw_weights, scale=7.0, base=0.5)
        else: 
            print("Error: No valid PMI matrix available.")
            return

    elif clustering_method == 4:
        if distance_metric in {"cosine",'default'} and similarity_matrix is not None: # TF-IDF cosine
            for i, word1 in enumerate(words):
                for j, word2 in enumerate(words):
                    if i != j:
                        similarity = similarity_matrix[i][j]
                        if similarity > 0.1:
                            G.add_edge(word1, word2, weight=similarity)
            raw_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_weights = normalize_edge_weights(raw_weights, scale=7.0, base=0.5)
        
        elif distance_metric == "raw_weighted" and co_occurrence_matrix is not None:  # TF-IDF Co-occurrence 
            for i, word1 in enumerate(words):
                for j, word2 in enumerate(words):
                    if i != j:
                        score = co_occurrence_matrix[i][j]
                        if score > 0.2:
                            G.add_edge(word1, word2, weight=score)
            raw_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_weights = normalize_edge_weights(raw_weights, scale=7.0, base=0.5)
        else:
            print("Error: No valid TF-IDF matrix available")
        
    else:
        print("Error: Invalid matrix input.")
        return
    
    # edge-weight normalisation 
    edge_list   = list(G.edges())
    raw_weights = [G[u][v]['weight'] for u,v in edge_list]

    base_width  = 0.35 * 5                  
    min_width   = base_width * 0.1         
    max_width   = base_width * 3.0          
    edge_weights=[]
    if raw_weights:
        mn,mx = min(raw_weights), max(raw_weights)
        edge_weights = [max_width if mx-mn==0 else
                        ((((w-mn)/(mx-mn))**0.7))*(max_width-min_width)+min_width
                        for w in raw_weights]

    visibility_threshold = (link_threshold if link_threshold is not None
                            else (0.2 if raw_weights else 0))
    strength_threshold   = (link_color_threshold if link_color_threshold is not None
                            else (np.median(raw_weights) if raw_weights else 0))

    # community layout
    communities={}
    for n in G.nodes():
        communities.setdefault(node_community_dict[n],[]).append(n)

    has_seed_community = "Seed" in communities

    community_pos = {}
    radius = 7.0                       
    angle_step = 2*np.pi/len(communities)
    for idx,comm in enumerate(communities):
        community_pos[comm]=(radius*np.cos(idx*angle_step),
                             radius*np.sin(idx*angle_step))

 

    # Set fixed seed for reproducibility
    import random  # Add this if not already imported
    np.random.seed(42)  # Ensure global randomness is fixed
    random.seed(42)     # Ensure random module is also fixed

    # Use kamada_kawai layout as the primary layout option
    if network_layout == "spring":
        pos_initial = nx_lib.spring_layout(G, k=1.0, iterations=300, seed=42)
    elif network_layout == "circular":
        pos_initial = nx_lib.circular_layout(G, scale=0.9)
    elif network_layout == "shell":
        pos_initial = nx_lib.shell_layout(G, scale=1.0, rotate=None)
    elif network_layout == "kamada-kawai":
        pos_initial = nx_lib.kamada_kawai_layout(G, scale=1.0, weight='weight')
    else:
        raise ValueError(f"Unknown network_layout: {network_layout}")

    # Adjust positions based on community structure
    pos_adjusted = {n: (0.2*pos_initial[n][0]+0.8*community_pos[node_community_dict[n]][0],
                   0.2*pos_initial[n][1]+0.8*community_pos[node_community_dict[n]][1])
                for n in G.nodes()}
    
    # # Ensure seed words are at least half the plot radius apart if there are two seed words:
    # if has_seed_community and len(communities["Seed"]) == 2:
    #     seed_words = communities["Seed"]
    #     min_distance = radius / 2  # Half the plot radius
    #     
    #     # Calculate current distance between seed words
    #     seed1, seed2 = seed_words[0], seed_words[1]
    #     current_distance = np.hypot(
    #         pos_adjusted[seed1][0] - pos_adjusted[seed2][0],
    #         pos_adjusted[seed1][1] - pos_adjusted[seed2][1]
    #     )
    #     
    #     # If seeds are too close, adjust their positions
    #     if current_distance < min_distance:
    #         # Calculate direction vector between seeds
    #         dx = pos_adjusted[seed2][0] - pos_adjusted[seed1][0]
    #         dy = pos_adjusted[seed2][1] - pos_adjusted[seed1][1]
    #         
    #         # Normalize direction vector
    #         norm = np.hypot(dx, dy)
    #         if norm > 0:
    #             dx, dy = dx/norm, dy/norm
    #         else:
    #             dx, dy = 1, 0  # Default direction if seeds are at same position
    #         
    #         # Move seeds apart in opposite directions
    #         adjustment = (min_distance - current_distance) / 2
    #         pos_adjusted[seed1] = (
    #             pos_adjusted[seed1][0] - dx * adjustment,
    #             pos_adjusted[seed1][1] - dy * adjustment
    #         )
    #         pos_adjusted[seed2] = (
    #             pos_adjusted[seed2][0] + dx * adjustment,
    #             pos_adjusted[seed2][1] + dy * adjustment
    #         )


    # relationship repulsion          
    # simple intra-community repulsion (identical)
    def repel_nodes(p,iterations=150,k=0.2):
        for _ in range(iterations):
            disp={n:[0,0] for n in p}
            for n1 in p:
                for n2 in p:
                    if n1==n2: continue
                    dx,dy=p[n1][0]-p[n2][0], p[n1][1]-p[n2][1]
                    dist=max(0.01,np.hypot(dx,dy))
                    if node_community_dict[n1]==node_community_dict[n2] and dist<0.7:
                        f=k/(dist*dist)
                        if dist<0.3: f*=3
                        disp[n1][0]+=dx*f/dist; disp[n1][1]+=dy*f/dist
            for n in p:
                dx,dy=disp[n]; mag=np.hypot(dx,dy)
                if mag>0.3: dx,dy=dx*0.3/mag, dy*0.3/mag
                p[n]=(p[n][0]+0.9*dx*1.5, p[n][1]+0.9*dy*1.5)
        return p
    node_positions = repel_nodes(pos_adjusted)

    # jitter for residual overlaps (unchanged)
    for n in node_positions:
        for other in node_positions:
            if n==other: continue
            if np.hypot(node_positions[n][0]-node_positions[other][0],
                        node_positions[n][1]-node_positions[other][1])<0.2:
                x,y=node_positions[n]
                node_positions[n]=(x+np.random.uniform(-0.2,0.2),
                                   y+np.random.uniform(-0.2,0.2)); break

    # edge categorisation
    edges_by_community_same, edges_by_community_diff = {}, {}
    edge_weights_by_community_same, edge_weights_by_community_diff = {}, {}
    strong_edges, strong_edge_weights=[],[]

    inter_community_edges, seed_to_community_edges = {}, {}

    for i,(u,v) in enumerate(edge_list):
        w=raw_weights[i]
        if w<visibility_threshold: continue
        cu,cv=node_community_dict[u], node_community_dict[v]
        key=tuple(sorted([cu,cv]))

        if has_seed_community:
            if cu=="Seed" and cv!="Seed":
                seed_to_community_edges.setdefault(cv,[]).append((u,v,w))
            elif cv=="Seed" and cu!="Seed":
                seed_to_community_edges.setdefault(cu,[]).append((v,u,w))

        if w>=strength_threshold and cu!=cv:
            strong_edges.append((u,v)); strong_edge_weights.append(edge_weights[i])
            inter_community_edges.setdefault(key,[]).append((u,v,w)); continue

        inter_community_edges.setdefault(key,[]).append((u,v,w))

        if cu==cv:
            edges_by_community_same .setdefault(key,[]).append((u,v))
            edge_weights_by_community_same.setdefault(key,[]).append(edge_weights[i])
        else:
            edges_by_community_diff .setdefault(key,[]).append((u,v))
            edge_weights_by_community_diff.setdefault(key,[]).append(edge_weights[i])

    # ensure at least one strong connection between seed and each category
    if has_seed_community and semantic_categories:
        for category in semantic_categories:
            if category == "Seed": continue
            
            # check if we already have a strong connection to this category
            key = tuple(sorted(["Seed", category]))
            has_strong_connection = False
            
            for u, v in strong_edges:
                cu, cv = node_community_dict[u], node_community_dict[v]
                if (cu == "Seed" and cv == category) or (cv == "Seed" and cu == category):
                    has_strong_connection = True
                    break
            
            # if no strong connection exists, add the strongest one
            if not has_strong_connection and category in seed_to_community_edges:
                strongest_edge = max(seed_to_community_edges[category], key=lambda x: x[2])
                u, v, w = strongest_edge
                
                # add to strong edges
                if (u, v) not in strong_edges and (v, u) not in strong_edges:
                    # calculate normalized edge weight
                    norm_weight = max_width if mx-mn==0 else ((w-mn)/(mx-mn))*(max_width-min_width)+min_width
                    strong_edges.append((u, v))
                    strong_edge_weights.append(norm_weight)

    # drawing order
    fig, ax = plt.subplots(figsize=(18,14))
    fig.patch.set_facecolor('white') 
    ax.set_facecolor('white')

    # inter-community (grey)  z=1  
    for key,edges in edges_by_community_diff.items():
        edge_color = "#888888" if "Seed" in key else "#cccccc"
        src_comm,tgt_comm = key
        if len(edges)>1:
            # community centroids for bundling
            src_nodes, tgt_nodes=set(),set()
            for u,v in edges:
                (src_nodes if node_community_dict[u]==src_comm else tgt_nodes).add(u)
                (tgt_nodes if node_community_dict[u]==src_comm else src_nodes).add(v)
            src_x = np.mean([node_positions[n][0] for n in src_nodes])
            src_y = np.mean([node_positions[n][1] for n in src_nodes])
            tgt_x = np.mean([node_positions[n][0] for n in tgt_nodes])
            tgt_y = np.mean([node_positions[n][1] for n in tgt_nodes])

            mid_x,mid_y=(src_x+tgt_x)/2,(src_y+tgt_y)/2
            dx,dy=tgt_x-src_x,tgt_y-src_y; ln=np.hypot(dx,dy)
            if ln>0:
                nx,ny = -dy/ln*0.5, dx/ln*0.5    # --- MOD --- bundling 0.3→0.5
                ctrl_x,ctrl_y = mid_x+nx, mid_y+ny
            for idx,(u,v) in enumerate(edges):
                w=edge_weights_by_community_diff[key][idx]
                pu,pv=node_positions[u], node_positions[v]
                ctrl1_x = pu[0]*0.65+ctrl_x*0.35; ctrl1_y = pu[1]*0.65+ctrl_y*0.35
                ctrl2_x = pv[0]*0.65+ctrl_x*0.35; ctrl2_y = pv[1]*0.65+ctrl_y*0.35
                path=Path([pu,(ctrl1_x,ctrl1_y),(ctrl_x,ctrl_y),
                           (ctrl2_x,ctrl2_y),pv],
                          [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.LINETO])
                ax.add_patch(patches.PathPatch(path,fc='none',ec=edge_color,
                                               lw=w*0.9,alpha=0.6,zorder=1))  # --- MOD z=1
        else:
            for idx,(u,v) in enumerate(edges):
                w=edge_weights_by_community_diff[key][idx]
                pu,pv=node_positions[u], node_positions[v]
                mid_x,mid_y=(pu[0]+pv[0])/2,(pu[1]+pv[1])/2
                dx,dy=pv[0]-pu[0], pv[1]-pu[1]; ln=np.hypot(dx,dy)
                if ln>0:
                    nx,ny=-dy/ln*0.2, dx/ln*0.2
                    ctrl_x,ctrl_y=mid_x+nx, mid_y+ny
                    path=Path([pu,(ctrl_x,ctrl_y),pv],
                              [Path.MOVETO,Path.CURVE3,Path.CURVE3])
                    ax.add_patch(patches.PathPatch(path,fc='none',ec=edge_color,
                                                   lw=w*0.9,alpha=0.6,zorder=1))  # --- MOD z=1
                else:
                    ax.plot([pu[0],pv[0]],[pu[1],pv[1]],
                            color=edge_color,lw=w*0.9,alpha=0.5,zorder=1)            # --- MOD z=1

    # intra-community (coloured) z=2 
    for key,edges in edges_by_community_same.items():
        edge_color = (semantic_categories[key[0]]['color']
                      if semantic_categories and key[0] in semantic_categories
                      else (seed_color if key[0]=="Seed" else default_color))
        for idx,(u,v) in enumerate(edges):
            w=edge_weights_by_community_same[key][idx]
            pu,pv=node_positions[u], node_positions[v]
            mid_x,mid_y=(pu[0]+pv[0])/2,(pu[1]+pv[1])/2
            dx,dy=pv[0]-pu[0], pv[1]-pu[1]; ln=np.hypot(dx,dy)
            if ln>0.1:
                nx,ny=-dy/ln*0.05, dx/ln*0.05
                ctrl_x,ctrl_y=mid_x+nx, mid_y+ny
                path=Path([pu,(ctrl_x,ctrl_y),pv],
                          [Path.MOVETO,Path.CURVE3,Path.CURVE3])
                ax.add_patch(patches.PathPatch(path,fc='none',ec=edge_color,
                                               lw=w*0.9,alpha=0.5,zorder=2))   
            else:
                ax.plot([pu[0],pv[0]],[pu[1],pv[1]],
                        color=edge_color,lw=w*0.9,alpha=0.5,zorder=2)            
    # strong cross-community (black) z=3 
    # (logic unchanged – only ensure zorder=3)
    unique_edges,unique_w=[],[]
    for i,e in enumerate(strong_edges):
        if e not in unique_edges and (e[1],e[0]) not in unique_edges:
            unique_edges.append(e); unique_w.append(strong_edge_weights[i])
    for i,(u,v) in enumerate(unique_edges):
        w=unique_w[i]; pu,pv=node_positions[u], node_positions[v]
        mid_x,mid_y=(pu[0]+pv[0])/2,(pu[1]+pv[1])/2
        dx,dy=pv[0]-pu[0], pv[1]-pu[1]; ln=np.hypot(dx,dy)
        curve_strength=0.15
        nx,ny=(-dy/ln*curve_strength, dx/ln*curve_strength) if ln else (0,0)
        ctrl_x,ctrl_y=mid_x+nx, mid_y+ny
        path=Path([pu,(ctrl_x,ctrl_y),pv],
                  [Path.MOVETO,Path.CURVE3,Path.CURVE3])
        ax.add_patch(patches.PathPatch(path,fc='none',ec='black',
                                       lw=w*0.9,alpha=1.0,zorder=3))
    # Add numeric value close to the edge for all strong cross-community links
    for i,(u,v) in enumerate(unique_edges):
        w = unique_w[i]
        pu, pv = node_positions[u], node_positions[v]
        
        # Calculate position for the label in the middle of the edge
        mid_x, mid_y = (pu[0]+pv[0])/2, (pu[1]+pv[1])/2
        
        # Add slight offset to avoid overlapping with the edge
        dx, dy = pv[0]-pu[0], pv[1]-pu[1]
        ln = np.hypot(dx, dy)
        if ln > 0:
            nx, ny = -dy/ln*0.1, dx/ln*0.1
            label_x, label_y = mid_x+nx, mid_y+ny
        else:
            label_x, label_y = mid_x, mid_y
        
        # Normalize the weight from 0.5-7.5 range to 1-5 range
        normalized_w = ((w - 0.5) / 7.0) * 4.0 + 1.0
        
        # Format normalized cosine similarity value to 2 decimal places
        weight_text = f"cos={normalized_w:.2f}"
        ax.text(label_x, label_y, weight_text, color='black', fontsize=10, 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", 
                fc='white', ec='none', alpha=0.8), zorder=4, fontweight='bold')

    # nodes & labels  (z=4 & z=5, unchanged)
    print(f"node colors is",node_color_dict)
    for n in G.nodes():
        x,y=node_positions[n]
        plt.scatter(x,y,s=3000,c=node_color_dict[n],
                    edgecolors='black',alpha=0.9,zorder=4)
    for n,(x,y) in node_positions.items():
        plt.text(x, y, n, fontsize=12, fontweight='bold', ha='center', va='center',
         color='black',
         bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black',
                   alpha=0.9), zorder=5)

    # legend & title (unchanged)    
    legend_handles=[]
    if semantic_categories:
        for lbl,grp in semantic_categories.items():
            if lbl in used_categories:
                legend_handles.append(
                    plt.Rectangle((0,0),1,1,color=grp['color'],label=lbl))
    if "Seed Word" in used_categories:
        legend_handles.append(
            plt.Rectangle((0,0),1,1,color=seed_color,label="Seed Word"))
    legend_handles.append(plt.Line2D([0],[0],color='black',lw=2,
                        label='Strongest Cross-Category Connections'))
    legend_handles.append(plt.Line2D([0],[0],color='#cccccc',lw=1.5,
                        label='Edge width = association strength'))

    plt.legend(handles=legend_handles,loc='upper right',
               bbox_to_anchor=(1.0,0.0),frameon=False, fontsize=16)
    plt.axis('off'); plt.tight_layout(); return fig

#---------------------------------------------------------------------------------
# MAIN PIPELINE FUNCTION
#---------------------------------------------------------------------------------

# Function to Run the Pipeline
def run_visuals_pipeline(input_data):
    """
    Main function to run the semantic network analysis pipeline.
    
    Parameters:
    -----------
    input_data : VisualsInput
        Object containing all input parameters
    """
    # Build reverse mapping for word families
    word_to_base = {}
    for base_word, variants in WORD_FAMILIES.items():
        for variant in variants:
            word_to_base[variant.lower()] = base_word

    seed_groups, seed_words, use_group_label = {}, [], False
    auto_selected_seeds = False

    df = pd.read_csv(input_data.filepath)
    
    # Normalize alternative column names if needed
    if 'text' not in df.columns:
        alternatives = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'body' in col.lower()]
        if alternatives:
            print(f"'text' column not found, using '{alternatives[0]}' instead.")
            df.rename(columns={alternatives[0]: 'text'}, inplace=True)
        else:
            print("Error: No suitable text column found.")
            return
    
    # We'll use the stop list instead of hardcoding additional common words
    additional_common_words = set()
            
    # Use the pre-loaded stopwords if available
    if hasattr(input_data, 'custom_stopwords') and input_data.custom_stopwords:
        stop_words = input_data.custom_stopwords
    else:
        # Load stop words the old way if not pre-loaded
        stop_words = manage_stop_list(input_data.stop_list, default_stop_words)
        # Add our additional common words to the stop list
        stop_words = stop_words.union(additional_common_words)

    
    # Fix list columns that may be stored as strings
    for col in ['data_group', 'codes']:
        if col in df.columns:
            # Check if first non-null value is a string that looks like a list
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, str) and (sample.startswith('[') or ',' in sample):
                df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and x.strip() else 
                                       ([] if pd.isna(x) else [x]))
    
    # Apply Metadata Filters
    if input_data.projects and 'project' in df.columns:
        before = len(df)
        df = df[df['project'].isin(input_data.projects)]
        print(f"Project filter: {before} → {len(df)} rows")
    
    if input_data.data_groups and 'data_group' in df.columns:
        before = len(df)
        try:
            # Handle both list and non-list data_group columns
            if df['data_group'].apply(lambda x: isinstance(x, list)).any():
                mask = df['data_group'].apply(lambda x: 
                    isinstance(x, list) and any(item in input_data.data_groups for item in x))
            else:
                mask = df['data_group'].isin(input_data.data_groups)
            df = df[mask]
            print(f"Data group filter: {before} → {len(df)} rows")
        except Exception as e:
            print(f"Error in data_group filtering: {e}")
            print(f"Sample data_group values: {df['data_group'].head()}")
    
    if input_data.codes and 'codes' in df.columns:
        before = len(df)
        try:
            # Handle both list and non-list codes columns
            if df['codes'].apply(lambda x: isinstance(x, list)).any():
                mask = df['codes'].apply(lambda x: 
                    isinstance(x, list) and any(item in input_data.codes for item in x))
            else:
                mask = df['codes'].isin(input_data.codes)
            df = df[mask]
            print(f"Codes filter: {before} → {len(df)} rows")
        except Exception as e:
            print(f"Error in codes filtering: {e}")
            print(f"Sample codes values: {df['codes'].head()}")
    
    if len(df) == 0:
        print("Error: All rows were filtered out. Please check your filter criteria.")
        return
    
    sentences = df['text'].dropna().tolist()
    print(f"Final dataset: {len(sentences)} text segments ready for processing")

    # Add filtered sentences tracking here
    filtered_sentences = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            print(f"Not a string: type={type(sentence)}, value={sentence}")
        tokens = tokenize_and_filter([sentence],stop_list=stop_words, 
                                   lemmatize=True, 
                                   cross_pos_normalize=input_data.cross_pos_normalize)
        if tokens:  # Only keep sentences that have tokens after filtering
            filtered_sentences.append(sentence)
    print(f"After filtering: {len(filtered_sentences)} valid text segments")
    

    # Filter out excluded codes if specified
    if hasattr(input_data, 'excluded_codes') and input_data.excluded_codes and 'codes' in df.columns:
        before = len(df)
        try:
            # Handle both list and non-list codes columns
            if df['codes'].apply(lambda x: isinstance(x, list)).any():
                # Keep rows where NONE of the excluded codes are present
                mask = df['codes'].apply(lambda x: 
                    isinstance(x, list) and not any(item in input_data.excluded_codes for item in x))
            else:
                # Keep rows where the code is not in excluded_codes
                mask = ~df['codes'].isin(input_data.excluded_codes)
            df = df[mask]
            print(f"Excluded codes filter: {before} → {len(df)} rows")
        except Exception as e:
            print(f"Error in excluded_codes filtering: {e}")
    
        # Define a custom word filter function
    def custom_word_filter(word):
        # First normalize with word families
        word_lower = word.lower()
        if word_lower in word_to_base:
            word = word_to_base[word_lower]
        
        # Manual exclusion of common words that should be filtered
        manual_exclusions = {'got', 'get', 'just', 'like', 'many', 'much', 'very', 'really', 'make'}
        
        return (word.lower() not in stop_words and
                word.lower() not in manual_exclusions and
                len(word) > 2 and  # Exclude very short words
                not any(c.isdigit() for c in word) and  # Exclude words with numbers
                re.match(r'^[a-z]+$', word.lower()))  # Only pure alphabetic words
                
    # Check if seed words were provided in the input
    if hasattr(input_data, 'seed_words') and input_data.seed_words and input_data.seed_words.strip().lower() != "none":
        seed_input = input_data.seed_words.strip()

        if ":" in seed_input:
            use_group_label = True
            for part in seed_input.split(";"):
                part = part.strip()
                if ":" in part:
                    group_label, word_str = part.split(":", 1)
                    group_label = group_label.strip().lower()
                    words = [w.strip().lower() for w in word_str.split(",") if w.strip()]
                    seed_groups[group_label] = set(words)
                    seed_words.append(group_label)
                    print(f"Group mode: all {words} will be treated as '{group_label}'")
                else:
                    individuals = [w.strip().lower() for w in part.split(",") if w.strip()]
                    seed_words.extend(individuals)
                    print(f"Individual mode: adding {individuals}")
        else:
            seed_words = [w.strip().lower() for w in seed_input.split(",") if w.strip()]
            print(f"Pure individual word mode: using {seed_words}")
        if use_group_label:
                sentences = [replace_group_words(text, seed_groups) for text in sentences]
    else:
        # Process sentences to get word frequencies for auto-selection of top words
        print("WARNING: No seed words provided or 'NONE' specified. Using top frequent words as seeds... ")
        excluded_words = stop_words.union(set(WORD_FAMILIES.keys()))
        all_tokens = []
        for sentence in sentences:
            tokens = tokenize_and_filter([sentence], stop_list=stop_words,
                                        lemmatize=True,
                                        cross_pos_normalize=input_data.cross_pos_normalize)
            filtered_tokens = [token.lower() for token in tokens if custom_word_filter(token)]
            all_tokens.extend(filtered_tokens)
        word_counts = Counter(all_tokens)
        top_words = [word for word, _ in word_counts.most_common(30)
                    if word.lower() not in excluded_words][:min(10, len(word_counts))]
        seed_words = top_words
        print(f"Top frequent words as seeds: {seed_words}")

    start = time.time()

    # Clean and normalize seed words, but they're already preprocessed with word families
    print(f"Original seed words before cleaning: {seed_words}")
    clean_seed_words = clean_words(seed_words)
    print(f"Seed words after cleaning: {clean_seed_words}")
    clean_seeds = clean_words(seed_words)
    seed_words = normalize_words(clean_seeds, stop_words, lemmatize=True, cross_pos_normalize=input_data.cross_pos_normalize)
    seed_words = list(set(seed_words))
    print("Final normalized seed words:", seed_words)
    
    # Check if any seed words remain after cleaning
    if not seed_words:
        print("WARNING: No seed words remain after processing. Using most frequent word as seed.")
        # Get most frequent word that's not in our expanded stop list
        for word, _ in word_counts.most_common(30):
            if custom_word_filter(word):
                top_word = word.lower()
                seed_words = [top_word]
                print(f"Using '{top_word}' as seed word.")
                break
    
    # choose context source: keep stop-words only for RoBERTa
    if input_data.clustering_method == 1:          # 1 = RoBERTa
        sentences_for_embedding = sentences        # full context
    else:                                          # 2-4 = Jaccard/PMI/TF-IDF
        sentences_for_embedding = filtered_sentences

    word_embeddings, similarity_matrix, co_occurrence_matrix = train_embedding(
        sentences_for_embedding,
        context_window = input_data.window_size, 
        stop_list  = stop_words, 
        seed_words = seed_words, 
        clustering_method  = input_data.clustering_method,
        num_words = input_data.num_words, 
        lemmatize = True, 
        min_word_frequency = input_data.min_word_frequency,
        reuse_clusterings  = input_data.reuse_clusterings,
        cross_pos_normalize= getattr(input_data, 'cross_pos_normalize', False),
        distance_metric = input_data.distance_metric,
        custom_word_filter = custom_word_filter
    )

    elapsed_time = time.time() - start

    # Plot Heatmap
    print("Plotting heatmap...")
    fig = plot_heatmap(
        input_data.clustering_method, word_embeddings, 
        similarity_matrix, co_occurrence_matrix, input_data.distance_metric
    )
    filename = f"heatmap_{input_data.clustering_method}_{input_data.distance_metric}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✔ [OK] Saved {out_path}")
    plt.show()


    # Generate t-SNE Dimensional Reduction Plot
    print("-----------------------------------------------")
    print("Generating t-SNE dimensional reduction plot...")
    try:
        plot_tsne_dimensional_reduction(
            word_embeddings=word_embeddings,
            similarity_matrix=similarity_matrix,
            co_occurrence_matrix=co_occurrence_matrix,
            clustering_method=input_data.clustering_method,
            seed_words=seed_words, 
            distance_metric=input_data.distance_metric,
        )
    except Exception as e:
        print(f"t-SNE plot error: {e}")

    print("-----------------------------------------------")
    print("Plotting semantic network (plain, no categories)...")
    fig_sn1 = plot_semantic_network(
        word_embeddings,
        [] if auto_selected_seeds else seed_words,
        input_data.clustering_method,
        similarity_matrix, co_occurrence_matrix,
        semantic_categories=None,               
        link_threshold = input_data.link_threshold,
        link_color_threshold= input_data.link_color_threshold,
        distance_metric=input_data.distance_metric)
    filename = f"semantic_network_plain_m{input_data.clustering_method}_{input_data.distance_metric}.png"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)

    fig_sn1.suptitle("Semantic Network (Plain)", fontsize=30, y=0.98, fontweight='bold')
    fig_sn1.subplots_adjust(top=0.95)

    fig_sn1.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✔ [OK] Saved {out_path}")

    plt.show()

    
    # Coloured visualisation (only if custom colors are enabled) 
    if getattr(input_data, 'custom_colors', False):
        # Print the actual list of words used in the network so users can match for coloring
        print("\n Word list available for coloring:")
        print(sorted(list(word_embeddings.keys())))
        
        print("\nUsing predefined semantic categories for custom grouping")
        # Use seed_words instead of empty list for the colored visualization
        fig_sn2 = plot_semantic_network(
            word_embeddings,
            seed_words,
            input_data.clustering_method,
            similarity_matrix, co_occurrence_matrix,
            semantic_categories = input_data.semantic_categories,
            link_threshold     = input_data.link_threshold,
            link_color_threshold = input_data.link_color_threshold,
            distance_metric=input_data.distance_metric,
        )
        filename = f"semantic_network_customcolor_m{input_data.clustering_method}_{input_data.distance_metric}.png"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, filename)

        fig_sn2.suptitle("Semantic Network (Custom Colors)", fontsize=30, y=0.98, fontweight='bold')
        fig_sn2.subplots_adjust(top=0.95)

        fig_sn2.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✔ [OK] Saved {out_path}")

        plt.show()


    # Add a second network visualization that removes seed nodes
    print("\nGenerating secondary network with seed nodes hidden...")
    # Filter out seed words from word embeddings and matrices
    non_seed_words = [word for word in word_embeddings.keys() if word not in seed_words]
    
    if len(non_seed_words) > 5:  # Only proceed if we have enough nodes to make a meaningful network
        non_seed_embeddings = {word: word_embeddings[word] for word in non_seed_words}
        
        # Create filtered similarity/co-occurrence matrices
        if similarity_matrix is not None:
            words = list(word_embeddings.keys())
            indices = [words.index(word) for word in non_seed_words]
            filtered_similarity = similarity_matrix[np.ix_(indices, indices)]
        else:
            filtered_similarity = None
            
        if co_occurrence_matrix is not None:
            words = list(word_embeddings.keys())
            indices = [words.index(word) for word in non_seed_words]
            filtered_co_occurrence = co_occurrence_matrix[np.ix_(indices, indices)]
        else:
            filtered_co_occurrence = None
        
        # Plot the filtered network
        print(f"Plotting secondary network with {len(non_seed_words)} nodes (seeds hidden)...")
        fig_sn3 = plot_semantic_network(
            non_seed_embeddings, [], 
            input_data.clustering_method, 
            filtered_similarity, filtered_co_occurrence, 
            semantic_categories=None,
            link_threshold=input_data.link_threshold,
            link_color_threshold=input_data.link_color_threshold,
            distance_metric=input_data.distance_metric,
        )
        filename = f"semantic_network_noseeds_m{input_data.clustering_method}_{input_data.distance_metric}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        fig_sn3.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✔ [OK] Saved {out_path}")

        fig_sn3.suptitle("Semantic Network (Seeds Hidden)", fontsize=30, y=0.98, fontweight='bold')
        fig_sn3.subplots_adjust(top=0.95)
        plt.show()
        
        # If custom coloring was used in the first visualization, also do it for the second
        if hasattr(input_data, 'custom_colors') and input_data.custom_colors and hasattr(input_data, 'semantic_categories'):
            semantic_categories = input_data.semantic_categories
            print("\nGenerating secondary network with custom grouping (seeds hidden)...")

            fig_sn4 = plot_semantic_network(
                non_seed_embeddings, [], 
                input_data.clustering_method, 
                filtered_similarity, filtered_co_occurrence, 
                semantic_categories=semantic_categories,
                link_threshold=input_data.link_threshold,
                link_color_threshold=input_data.link_color_threshold,
                distance_metric=input_data.distance_metric,
            )
            filename = f"semantic_network_noseeds_customcolor_m{input_data.clustering_method}_{input_data.distance_metric}.png"
            out_path = os.path.join(OUTPUT_DIR, filename)
            fig_sn4.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"✔ [OK] Saved {out_path}")

            fig_sn4.suptitle("Semantic Network with Custom Grouping (Seeds Hidden)", fontsize=30, y=0.98, fontweight='bold')
            fig_sn4.subplots_adjust(top=0.95)
            plt.show()
    else:
        print("Not enough non-seed nodes to generate a meaningful secondary network.")
    
    # Print the list of actual nodes used in the network (excluding seeds)
    print("\nFinal nodes used in network (excluding seeds):")
    if 'non_seed_words' in locals() and len(non_seed_words) > 0:
        # Get frequencies for each node and sort by frequency (highest first)
        print(f"Total non-seed words: {len(non_seed_words)}")
        
        # Check if word_frequencies is defined, otherwise use the word_counts from earlier
        if 'word_frequencies' not in locals() and 'word_counts' in locals():
            word_frequencies = word_counts
        elif 'word_frequencies' not in locals():
            print("Warning: Word frequency information not available")
            # Just print the words without frequencies
            for word in sorted(non_seed_words):
                print(f"- {word}")
        else:
            # Create list of (word, frequency) pairs and sort by frequency
            freq_sorted_words = [(word, word_frequencies[word]) for word in non_seed_words if word in word_frequencies]
            freq_sorted_words.sort(key=lambda x: x[1], reverse=True)
            
            for word, freq in freq_sorted_words:
                print(f"- {word}: {freq}")
    else:
        print("No non-seed nodes were used in the network.uelse")
    
    # Print the number of filtered sentences used
    if 'filtered_sentences' in locals():
        print(f"\nTotal number of filtered sentences used: {len(filtered_sentences)}")
        if 'seed_words' in locals() and seed_words:
            # Enhanced seed word detection using word families
            seed_containing_sentences = 0
            for sentence in filtered_sentences:
                sentence_lower = sentence.lower()
                contains_seed = False
                for seed in seed_words:
                    # Check direct match
                    if seed.lower() in sentence_lower:
                        contains_seed = True
                        break
                    # Check word family variants
                    for family, variants in WORD_FAMILIES.items():
                        if seed.lower() == family.lower() or seed.lower() in [v.lower() for v in variants]:
                            if any(variant.lower() in sentence_lower for variant in variants):
                                contains_seed = True
                                break
                    if contains_seed:
                        break
                if contains_seed:
                    seed_containing_sentences += 1
            print(f"Number of filtered sentences containing seed words: {seed_containing_sentences}")
    
    print("\nAnalysis complete")
    print(f"\nNetwork visualization method: {input_data.clustering_method}")
    
    if input_data.clustering_method == 1:
        print("Method: RoBERTa – Shows semantic relationships based on contextual embeddings")
    elif input_data.clustering_method == 2:
        if input_data.distance_metric == "cosine":
            print("Method: Jaccard (cosine) – Uses context window vectors to compute cosine-based similarity between word usage patterns")
        elif input_data.distance_metric == "default":
            print("Method: Jaccard (default) – Uses binary co-occurrence counts within a context window to capture word overlap")
    elif input_data.clustering_method == 3:
        if input_data.distance_metric == "cosine":
            print("Method: PMI (cosine) – Highlights statistically significant word associations using cosine similarity of PMI-weighted context vectors")
    elif input_data.clustering_method == 4:
        if input_data.distance_metric in {'cosine', 'default'}:
            print("Method: TF-IDF (cosine) – Uses TF-IDF-weighted context vectors to compute cosine similarity between words")
        elif input_data.distance_metric == "raw_weighted":
            print("Method: TF-IDF (raw weighted) – Uses raw TF-IDF-weighted co-occurrence scores for word associations")

# STOP Programs

def ask_yes_no(prompt, default=True):
    hint  = " [Y/n]" if default else " [y/N]"
    while True:
        try:
            s = input(prompt + hint + " (q=quit) ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled."); raise SystemExit(0)
        if s == "": return default            # Enter → default
        if s in {"y","yes"}: return True
        if s in {"n","no"}:  return False
        if s in {"q","quit"}: raise SystemExit(0)
        print("Please answer y/n (or q to quit).")

def ask_path(prompt, default_path):
    while True:
        try:
            s = input(f"{prompt} (Enter for default: {default_path}, q=quit) ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled."); raise SystemExit(0)
        if s.lower() in {"q","quit"}: raise SystemExit(0)  
        if s == "": return default_path                   
        return s
