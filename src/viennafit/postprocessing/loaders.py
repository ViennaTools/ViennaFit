"""
Data loading utilities for optimization and GSS results.
"""

import os
import json
import ast
import csv
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from .base import StudyData


class ResultsLoader:
    """Unified data loader for both optimization and GSS results."""
    
    def __init__(self, runDir: str):
        self.runDir = os.path.abspath(runDir)
        self.studyName = os.path.basename(self.runDir)
        
    def loadStudyData(self) -> StudyData:
        """Load all available data for this study."""
        studyType = self._detectStudyType()
        
        data = StudyData(
            runDir=self.runDir,
            studyName=self.studyName,
            studyType=studyType
        )
        
        # Load common data types
        data.metadata = self._loadMetadata()
        data.progressData = self._loadProgressData()
        data.results = self._loadResults()
        
        # Load study-type specific data
        if studyType == 'global_sensitivity':
            data.evaluationData = self._loadEvaluationResults()
            data.sensitivityData = self._loadSensitivityResults()
            
        return data
        
    def _detectStudyType(self) -> str:
        """Detect whether this is an optimization or GSS study."""
        # Check for GSS-specific files
        if (os.path.exists(os.path.join(self.runDir, 'sensitivity_results.json')) or
            os.path.exists(os.path.join(self.runDir, 'parameter_samples.csv')) or
            os.path.exists(os.path.join(self.runDir, 'evaluations'))):
            return 'global_sensitivity'
        else:
            return 'optimization'
            
    def _loadMetadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata from JSON file."""
        metadataPaths = [
            os.path.join(self.runDir, 'progressAll_metadata.json'),
            os.path.join(self.runDir, f'{self.studyName}_metadata.json'),
            os.path.join(self.runDir, 'metadata.json')
        ]
        
        for path in metadataPaths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load metadata from {path}: {e}")
                    
        return None
        
    def _loadResults(self) -> Optional[Dict[str, Any]]:
        """Load optimization results from JSON files."""
        resultPatterns = [
            f'{self.studyName}-final-results.json',
            'results.json',
            'bestResult.json',
            'Results.json'
        ]
        
        for pattern in resultPatterns:
            resultPath = os.path.join(self.runDir, pattern)
            if os.path.exists(resultPath):
                try:
                    with open(resultPath, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load results from {resultPath}: {e}")
                    
        return None
        
    def _loadProgressData(self) -> Optional[Dict[str, Any]]:
        """Load progress data from CSV or legacy TXT files."""
        progressData = {}
        
        # Try CSV format first (new format)
        csvPaths = [
            ('all', os.path.join(self.runDir, 'progressAll.csv')),
            ('best', os.path.join(self.runDir, 'progressAll_best.csv'))
        ]
        
        for dataType, path in csvPaths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    progressData[dataType] = df
                except Exception as e:
                    print(f"Warning: Could not load CSV progress data from {path}: {e}")
                    
        # Try legacy TXT format if CSV not available
        if not progressData:
            txtPaths = [
                ('all', os.path.join(self.runDir, 'progressAll.txt')),
                ('best', os.path.join(self.runDir, 'progress.txt'))
            ]
            
            for dataType, path in txtPaths:
                if os.path.exists(path):
                    try:
                        data = self._parseLegacyProgressFile(path)
                        if data is not None and len(data) > 0:
                            progressData[dataType] = data
                    except Exception as e:
                        print(f"Warning: Could not load TXT progress data from {path}: {e}")
                        
        return progressData if progressData else None
        
    def _parseLegacyProgressFile(self, filePath: str) -> Optional[np.ndarray]:
        """Parse legacy progress file containing Python list strings."""
        try:
            with open(filePath, 'r') as f:
                lines = f.readlines()
                
            dataRows = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        dataList = ast.literal_eval(line)
                        if isinstance(dataList, list):
                            dataRow = [float(x) for x in dataList]
                            dataRows.append(dataRow)
                    except (ValueError, SyntaxError):
                        continue
                        
            if dataRows:
                # Handle variable column counts by padding
                maxCols = max(len(row) for row in dataRows)
                paddedRows = []
                for row in dataRows:
                    if len(row) < maxCols:
                        row.extend([np.nan] * (maxCols - len(row)))
                    paddedRows.append(row)
                    
                return np.array(paddedRows)
                
        except Exception as e:
            print(f"Error parsing legacy progress file {filePath}: {e}")
            
        return None
        
    def _loadEvaluationResults(self) -> Optional[List[Dict[str, Any]]]:
        """Load GSS evaluation results."""
        evalPath = os.path.join(self.runDir, 'evaluation_results.json')
        if os.path.exists(evalPath):
            try:
                with open(evalPath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load evaluation results: {e}")
                
        return None
        
    def _loadSensitivityResults(self) -> Optional[Dict[str, Any]]:
        """Load GSS sensitivity analysis results."""
        sensPath = os.path.join(self.runDir, 'sensitivity_results.json')
        if os.path.exists(sensPath):
            try:
                with open(sensPath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load sensitivity results: {e}")
                
        return None
        
    def getParameterNames(self) -> List[str]:
        """Extract parameter names from available data."""
        # Try metadata first
        metadata = self._loadMetadata()
        if metadata and 'parameterNames' in metadata:
            return metadata['parameterNames']
            
        # Try progress data CSV headers
        progressData = self._loadProgressData()
        if progressData and 'all' in progressData:
            df = progressData['all']
            if hasattr(df, 'columns'):
                # Filter out non-parameter columns
                nonParamCols = {'evaluationNumber', 'elapsedTime', 'objectiveValue'}
                paramCols = [col for col in df.columns if col not in nonParamCols]
                return paramCols
                
        # Try results file
        results = self._loadResults()
        if results and 'bestParameters' in results:
            return list(results['bestParameters'].keys())
            
        # Try GSS evaluation results
        evalResults = self._loadEvaluationResults()
        if evalResults and len(evalResults) > 0:
            firstResult = evalResults[0]
            if 'parameters' in firstResult:
                return list(firstResult['parameters'].keys())
                
        return []
        
    def getConvergenceData(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract convergence data (evaluation numbers, objective values)."""
        progressData = self._loadProgressData()
        if not progressData:
            return None, None
            
        # Try 'all' data first
        if 'all' in progressData:
            data = progressData['all']
            if hasattr(data, 'values'):  # pandas DataFrame
                evalNums = data['evaluationNumber'].values
                objVals = data['objectiveValue'].values
            else:  # numpy array
                evalNums = np.arange(1, len(data) + 1)
                objVals = data[:, -1]  # Last column is objective value
            return evalNums, objVals
            
        # Fall back to 'best' data
        if 'best' in progressData:
            data = progressData['best']
            if hasattr(data, 'values'):  # pandas DataFrame
                evalNums = data['evaluationNumber'].values
                objVals = data['objectiveValue'].values
            else:  # numpy array
                evalNums = np.arange(1, len(data) + 1)
                objVals = data[:, -1]  # Last column is objective value
            return evalNums, objVals
            
        return None, None