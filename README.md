# MSPAT - MultiScale Protein Analysis Toolkit

A Python toolkit for protein analysis and machine learning applications, designed as a monorepo workspace containing specialized packages for different aspects of protein data processing.

WORK IN PROGRESS: This project is under active development and is not fully functional yet.

## Overview

MSPAT provides a complete pipeline from raw protein data to machine learning applications through three core packages:

- **padata**: Core data infrastructure with tensor-based data structures and transformation pipelines
- **biocontacts**: Molecular interaction analysis toolkit for hydrogen bonds, hydrophobic interactions, and more
- **padatasets**: Machine learning integration with dataset utilities and task implementations
- **osif**: Additional models and training utilities for protein analysis

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install from source

```bash
git clone https://github.com/ilbumi/mspat.git
cd mspat
uv sync
```

This will install all packages in the workspace along with their dependencies.

## Package Details

### padata - Core Data Infrastructure

Provides the foundation for protein data handling:

- **Tensor Classes**: Standardized data structures for protein sequences and structures
- **Transform Pipeline**: Modular preprocessing components including bond addition and protonation handling
- **Vocabulary Management**: Residue-level vocabulary and encoding systems
- **Utilities**: Padding and data manipulation tools

### biocontacts - Molecular Interaction Analysis

Specialized toolkit for analyzing molecular interactions:

- **Interaction Types**: Hydrogen bonds, hydrophobic, ionic, pi-interactions, and steric clashes
- **Descriptors**: Some descriptors for intermolecular interactions

### padatasets - Machine Learning Integration (WIP)

Dataset utilities and task implementations for protein ML:

- **Dataset Utilities**: Tools for creating ML-ready datasets from protein data
- **Task Implementations**: Sequence reconstruction tasks and structure prediction workflows

### osif - Old models for Inverse Foldings, must be refactored

- **Models**: OSIF models for inverse folding tasks
- **Training Utilities**: Training loops and utilities for OSIF models

## Development

### Common Commands

```bash
# Run tests with coverage
make test

# Lint code
make lint

# Format code
make format

# Build documentation
make docs

# Sync dependencies
make sync
```

### Code Quality

The project uses comprehensive code quality tools:

- **Linting**: `ruff` with extensive rule set and `mypy` for type checking
- **Formatting**: `ruff format`, `isort`, and `ssort` for consistent code style
- **Testing**: `pytest` with coverage reporting and parallel execution

## Testing

Run the complete test suite:

```bash
make test
```

Or run tests for specific packages:

```bash
uv run pytest tests/padata/
uv run pytest tests/biocontacts/
```

## Documentation

Build the documentation:

```bash
make docs
```

The documentation is built using Sphinx and includes API references with type hints.

## Contributing

1. Ensure all tests pass: `make test`
2. Check code quality: `make lint`
3. Format code: `make format`
4. Update documentation as needed
