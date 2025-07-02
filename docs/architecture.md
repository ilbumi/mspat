# Architecture

## Overview

MSPAT (MultiScale Protein Analysis Toolkit) is designed as a monorepo workspace containing several core packages that work together to provide a complete pipeline from raw protein data to machine learning applications.

## Package Architecture

### padata - Core Data Infrastructure
The foundation package providing tensor-based data structures and transformation pipelines:

- **Tensor Classes**: Standardized data structures for protein sequences and structures
- **Transform Pipeline**: Modular preprocessing components including:
  - Bond addition and analysis
  - Protonation state handling
  - Composable transformation chains
- **Vocabulary Management**: Residue-level vocabulary and encoding
- **Utilities**: Padding and data manipulation tools

### biocontacts - Molecular Interaction Analysis
Specialized toolkit for analyzing molecular interactions:

- **Interaction Types**:
  - Hydrogen bonds
  - Hydrophobic interactions
  - Ionic interactions
  - Pi-interactions (pi-pi, cation-pi)
  - Steric clashes
- **Interface Analysis**: Tools for protein-protein interface characterization
- **Descriptors**: Common geometric and chemical descriptors for interactions

### padatasets - Machine Learning Integration
Dataset utilities and task implementations for protein ML:

- **Dataset Utilities**: Tools for creating ML-ready datasets from protein data
- **Task Implementations**: Specific ML tasks including:
  - Sequence reconstruction tasks
  - Structure prediction workflows
- **Data Loaders**: Efficient data loading for training pipelines
