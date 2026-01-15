# Setup Examples

This folder contains examples for setting up ViennaFit projects and creating initial domains.

## Files

### 1-initializeProject.py
- Creates and initializes a new ViennaFit project
- Shows different initialization options
- Creates the basic project structure

### 2-createInitialDomain.py
- Loads annotation data from the data folder
- Creates initial and target level set domains
- Sets up domains for optimization and analysis

## Usage

Run these examples in order:

1. First run `initializeProject.py` to create the project structure
2. Copy annotation files from `../data/` to the project's `domains/annotations/` directory
3. Run `createInitialDomain.py` to set up the domains

After completing these steps, you can proceed to the optimization and sensitivity analysis examples.
