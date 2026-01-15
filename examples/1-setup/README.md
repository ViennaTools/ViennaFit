# Setup Examples

This folder contains examples for setting up ViennaFit projects and creating initial domains.

## Files

### initializeProject.py
- Creates and initializes a new ViennaFit project
- Copies annotation files from `../0-example-data/` to the project's `domains/annotations/` folder
- Creates the basic project structure

### assignDomains.py
- Loads annotation data from the project's `domains/annotations/` folder
- Creates initial and target level set domains
- Sets up domains for optimization and analysis

## Usage

Run these examples in order:

1. First run `initializeProject.py` to create the project structure and copy annotation files
2. Run `assignDomains.py` to set up the domains

After completing these steps, you can proceed to the optimization and sensitivity analysis examples.
