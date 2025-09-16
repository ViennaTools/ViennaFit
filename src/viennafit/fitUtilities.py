import json
import numpy as np
import time
import os
import csv
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


def readPointsFromFile(
    filename,
    mesh,
    gridDelta,
    minx=1e100,
    maxx=-1e100,
    miny=1e100,
    maxy=-1e100,
    minz=1e100,
    maxz=-1e100,
    scaleFactor=1,
    shiftX=0,
    shiftY=0,
    shiftZ=0,
    mode="3D",
    reflectX=False,
):
    try:
        with open(filename, "r") as file:
            for line in file:
                if mode == "3D":
                    x, y, z = map(float, line.strip().split())
                elif mode == "2D":
                    x, y = map(float, line.strip().split())
                    z = 0.0  # Default z value for 2D mode

                if reflectX:
                    x = -x

                minx = min(minx, x)
                miny = min(miny, y)
                minz = min(minz, z)
                maxx = max(maxx, x)
                maxy = max(maxy, y)
                maxz = max(maxz, z)
                idx = mesh.insertNextNode(
                    [
                        x * scaleFactor + shiftX,
                        y * scaleFactor + shiftY,
                        z * scaleFactor + shiftZ,
                    ]
                )
                if idx > 0:
                    mesh.insertNextLine([idx - 1, idx])
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
    except ValueError as e:
        print(f"Error processing line: {e}")

    minx = minx * scaleFactor + shiftX
    miny = miny * scaleFactor + shiftY
    minz = minz * scaleFactor + shiftZ
    maxx = maxx * scaleFactor + shiftX
    maxy = maxy * scaleFactor + shiftY
    maxz = maxz * scaleFactor + shiftZ

    if mode == "2D":
        extent = [
            gridDelta * round(minx / gridDelta),
            gridDelta * round(maxx / gridDelta),
            gridDelta * round(miny / gridDelta),
            gridDelta * round(maxy / gridDelta),
        ]
        return extent
    else:
        extent = [minx, maxx, miny, maxy, minz, maxz]
        # round the extent to the nearest gridDelta
        extent = [
            gridDelta * round(extent[0] / gridDelta),
            gridDelta * round(extent[1] / gridDelta),
            gridDelta * round(extent[2] / gridDelta),
            gridDelta * round(extent[3] / gridDelta),
            gridDelta * round(extent[4] / gridDelta),
            gridDelta * round(extent[5] / gridDelta),
        ]
        return extent


def saveDataToJson(data, filename, indent=2):
    """
    Save data to a JSON file with better formatting and structure.

    Args:
        data: Dictionary of data to save
        filename: Path to save the JSON file
        indent: Number of spaces for indentation (default: 2)
    """
    # Convert NumPy arrays and other non-serializable types
    data_serializable = {}

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # For arrays, preserve shape information
            data_serializable[key] = {
                "type": "ndarray",
                "shape": value.shape,
                "dtype": str(value.dtype),
                "data": value.tolist(),
            }
        elif isinstance(value, np.number):
            # Handle NumPy scalar types
            data_serializable[key] = float(value)
        else:
            data_serializable[key] = value

    # Add metadata
    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": f"Output data for {filename}",
    }

    output = {"metadata": metadata, "data": data_serializable}

    with open(filename, "w") as file:
        json.dump(output, file, indent=indent)

    print(f"Data saved to {filename}")


def loadDataFromJson(filename):
    """Load data from JSON, reconstructing NumPy arrays if present"""
    with open(filename, "r") as file:
        loaded = json.load(file)

    # Extract the data section
    data = loaded["data"]

    # Reconstruct NumPy arrays
    for key, value in data.items():
        if isinstance(value, dict) and "type" in value and value["type"] == "ndarray":
            data[key] = np.array(value["data"])

    return data, loaded["metadata"]


def saveEvalToProgressFile(evaluation, filepath):
    """Legacy function for backward compatibility"""
    # limit the number of decimal places to 5
    evaluation = [round(i, 5) for i in evaluation]
    evaluation = [str(i) for i in evaluation]

    with open(filepath, "a") as file:
        file.write(str(evaluation) + "\n")


def saveEvalToProgressManager(
    progressManager: 'ProgressDataManager',
    evaluationNumber: int,
    parameterValues: List[float],
    elapsedTime: float,
    objectiveValue: float,
    isBest: bool = False,
    saveAll: bool = True
) -> None:
    """Save evaluation using the new progress manager system"""
    record = EvaluationRecord(
        evaluationNumber=evaluationNumber,
        parameterValues=parameterValues,
        elapsedTime=elapsedTime,
        objectiveValue=objectiveValue,
        isBest=isBest
    )
    
    progressManager.saveEvaluation(record, saveAll=saveAll)


def getBestParams(filename):
    with open(filename) as file:
        lines = file.readlines()
    # find the line with the best objective function value
    bestLine = ""
    bestObjFunc = 1e10
    for line in lines:
        objFunc = float(line.split(",")[-1].strip().strip("[]'"))
        if objFunc < bestObjFunc:
            bestObjFunc = objFunc
            bestLine = line
    # extract the parameters from the best line
    bestParams = bestLine.split(",")
    bestParams = [float(i.strip().strip("[]'")) for i in bestParams]
    # drop the last two elements (time and objective function value)
    bestParams = bestParams[:-2]

    return bestParams


def loadProgressFileArrays(filename):
    with open(filename) as file:
        lines = file.readlines()
    # convert the lines to a numpy array
    data = []
    for line in lines:
        data.append([float(i.strip().strip("[]'")) for i in line.split(",")])
    # Convert to numpy array
    dataArray = np.array(data)
    # Extract all parameters (excluding the last two columns which are time and objective function)
    X = dataArray[:, :-2]
    # Extract objective function values (last column)
    Y = dataArray[:, -1]
    return X, Y


def loadOptimumFromResultsFile(filename):
    """
    Load the optimum parameters and results from a JSON results file.

    Args:
        filename (str): Path to the JSON results file.

    Returns:
        dict: Dictionary containing the optimization results including:
            - bestParameters: Dictionary of optimized parameter values
            - bestScore: Best objective function value achieved
            - bestEvaluation#: Evaluation number where best result was found
            - fixedParameters: Dictionary of fixed parameters
            - variableParameters: Dictionary of variable parameter bounds
            - optimizer: Name of optimizer used
            - numEvaluations: Total number of evaluations performed
    """
    if not filename.endswith(".json"):
        raise ValueError("Results file must be in JSON format (.json extension)")

    try:
        with open(filename, "r") as file:
            results = json.load(file)

        # Validate that required fields are present
        required_fields = ["bestParameters", "bestScore"]
        for field in required_fields:
            if field not in results:
                raise KeyError(f"Required field '{field}' not found in results file")

        return results

    except FileNotFoundError:
        raise FileNotFoundError(f"Results file '{filename}' not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in results file: {e}")


def getOptimumParameters(filename):
    """
    Extract only the optimum parameter values from a results file.

    Args:
        filename (str): Path to the JSON results file.

    Returns:
        dict: Dictionary containing only the best parameter values.
    """
    results = loadOptimumFromResultsFile(filename)
    return results["bestParameters"]


def getOptimumScore(filename):
    """
    Extract only the best objective function value from a results file.

    Args:
        filename (str): Path to the JSON results file.

    Returns:
        float: Best objective function value achieved.
    """
    results = loadOptimumFromResultsFile(filename)
    return results["bestScore"]


@dataclass
class ProgressMetadata:
    """Metadata for progress tracking"""
    runName: str
    parameterNames: List[str]
    parameterBounds: Dict[str, Tuple[float, float]]
    fixedParameters: Dict[str, float]
    optimizer: str
    createdTime: str
    description: str = ""
    
    def toDict(self) -> Dict[str, Any]:
        return {
            "runName": self.runName,
            "parameterNames": self.parameterNames,
            "parameterBounds": self.parameterBounds,
            "fixedParameters": self.fixedParameters,
            "optimizer": self.optimizer,
            "createdTime": self.createdTime,
            "description": self.description
        }
    
    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> 'ProgressMetadata':
        return cls(**data)


@dataclass
class EvaluationRecord:
    """Single evaluation record"""
    evaluationNumber: int
    parameterValues: List[float]
    elapsedTime: float
    objectiveValue: float
    isBest: bool = False
    
    def toDict(self) -> Dict[str, Any]:
        return {
            "evaluationNumber": self.evaluationNumber,
            "parameterValues": self.parameterValues,
            "elapsedTime": self.elapsedTime,
            "objectiveValue": self.objectiveValue,
            "isBest": self.isBest
        }


class ProgressDataManager(ABC):
    """Abstract base class for progress data storage"""
    
    def __init__(self, filepath: str, metadata: Optional[ProgressMetadata] = None):
        self.filepath = filepath
        self.metadata = metadata
        self.bestRecords: List[EvaluationRecord] = []
        self.allRecords: List[EvaluationRecord] = []
    
    @abstractmethod
    def saveEvaluation(self, record: EvaluationRecord, saveAll: bool = True) -> None:
        """Save a single evaluation record"""
        pass
    
    @abstractmethod
    def loadData(self) -> Tuple[List[EvaluationRecord], List[EvaluationRecord], ProgressMetadata]:
        """Load all data and return (bestRecords, allRecords, metadata)"""
        pass
    
    @abstractmethod
    def saveMetadata(self) -> None:
        """Save metadata to file"""
        pass
    
    def getBestRecords(self) -> List[EvaluationRecord]:
        """Get all best evaluation records"""
        return self.bestRecords
    
    def getAllRecords(self) -> List[EvaluationRecord]:
        """Get all evaluation records"""
        return self.allRecords
    
    def getConvergenceData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get convergence data as (evaluationNumbers, objectiveValues)"""
        if not self.bestRecords:
            return np.array([]), np.array([])
        
        evalNums = np.array([r.evaluationNumber for r in self.bestRecords])
        objVals = np.array([r.objectiveValue for r in self.bestRecords])
        return evalNums, objVals
    
    def getAllConvergenceData(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all evaluations convergence data"""
        if not self.allRecords:
            return np.array([]), np.array([])
        
        evalNums = np.array([r.evaluationNumber for r in self.allRecords])
        objVals = np.array([r.objectiveValue for r in self.allRecords])
        return evalNums, objVals
    
    def getParameterData(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get parameter data for best solution"""
        if not self.bestRecords:
            return np.array([]), {}
        
        bestRecord = min(self.bestRecords, key=lambda r: r.objectiveValue)
        paramArray = np.array(bestRecord.parameterValues)
        
        if self.metadata:
            paramDict = dict(zip(self.metadata.parameterNames, bestRecord.parameterValues))
        else:
            paramDict = {f"param_{i}": val for i, val in enumerate(bestRecord.parameterValues)}
        
        return paramArray, paramDict


class CSVProgressStorage(ProgressDataManager):
    """CSV-based progress storage with metadata support"""
    
    def __init__(self, filepath: str, metadata: Optional[ProgressMetadata] = None):
        if not filepath.endswith('.csv'):
            # If filepath is a directory or doesn't have extension, add .csv
            if os.path.isdir(filepath) or not os.path.splitext(filepath)[1]:
                filepath = os.path.join(filepath, 'progressAll.csv') if os.path.isdir(filepath) else filepath + '.csv'
            else:
                filepath = filepath.replace('.txt', '.csv')
        super().__init__(filepath, metadata)
        # Generate progressBest.csv instead of progressAll_best.csv for cleaner naming
        base_dir = os.path.dirname(filepath)
        self.bestFilepath = os.path.join(base_dir, 'progressBest.csv')
        self.metadataFilepath = filepath.replace('.csv', '_metadata.json')
    
    def saveEvaluation(self, record: EvaluationRecord, saveAll: bool = True) -> None:
        """Save evaluation record to CSV files"""
        # Save to all evaluations file
        if saveAll:
            self._saveRecordToCsv(record, self.filepath)
            self.allRecords.append(record)
        
        # Save to best evaluations file if it's a best record
        if record.isBest:
            self._saveRecordToCsv(record, self.bestFilepath)
            self.bestRecords.append(record)
    
    def _saveRecordToCsv(self, record: EvaluationRecord, filepath: str) -> None:
        """Save a single record to CSV file"""
        fileExists = os.path.exists(filepath)
        
        with open(filepath, 'a', newline='') as csvfile:
            # Use actual parameter count instead of metadata names to avoid index errors
            actualParamCount = len(record.parameterValues)
            
            if self.metadata and len(self.metadata.parameterNames) == actualParamCount:
                fieldnames = ['evaluationNumber'] + self.metadata.parameterNames + ['elapsedTime', 'objectiveValue']
            else:
                fieldnames = ['evaluationNumber'] + [f'param_{i}' for i in range(actualParamCount)] + ['elapsedTime', 'objectiveValue']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not fileExists:
                writer.writeheader()
            
            # Write data row
            row = {'evaluationNumber': record.evaluationNumber}
            paramFieldnames = fieldnames[1:-2]  # Skip evalNumber and last two columns
            for i, paramName in enumerate(paramFieldnames):
                if i < len(record.parameterValues):
                    row[paramName] = record.parameterValues[i]
                else:
                    row[paramName] = 0.0  # Default value if not enough parameters
            row['elapsedTime'] = record.elapsedTime
            row['objectiveValue'] = record.objectiveValue
            
            writer.writerow(row)
    
    def loadData(self) -> Tuple[List[EvaluationRecord], List[EvaluationRecord], ProgressMetadata]:
        """Load data from CSV files"""
        # Load metadata
        metadata = self._loadMetadata()
        
        # Load all records
        allRecords = []
        if os.path.exists(self.filepath):
            allRecords = self._loadRecordsFromCsv(self.filepath)
        
        # Load best records
        bestRecords = []
        if os.path.exists(self.bestFilepath):
            bestRecords = self._loadRecordsFromCsv(self.bestFilepath)
        
        self.allRecords = allRecords
        self.bestRecords = bestRecords
        self.metadata = metadata
        
        return bestRecords, allRecords, metadata
    
    def _loadRecordsFromCsv(self, filepath: str) -> List[EvaluationRecord]:
        """Load records from a CSV file"""
        records = []
        
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                evalNum = int(row['evaluationNumber'])
                elapsedTime = float(row['elapsedTime'])
                objectiveValue = float(row['objectiveValue'])
                
                # Extract parameter values
                paramValues = []
                for key, value in row.items():
                    if key not in ['evaluationNumber', 'elapsedTime', 'objectiveValue']:
                        paramValues.append(float(value))
                
                record = EvaluationRecord(
                    evaluationNumber=evalNum,
                    parameterValues=paramValues,
                    elapsedTime=elapsedTime,
                    objectiveValue=objectiveValue,
                    isBest=(filepath == self.bestFilepath)
                )
                records.append(record)
        
        return records
    
    def saveMetadata(self) -> None:
        """Save metadata to JSON file"""
        if self.metadata:
            with open(self.metadataFilepath, 'w') as f:
                json.dump(self.metadata.toDict(), f, indent=2)
    
    def _loadMetadata(self) -> Optional[ProgressMetadata]:
        """Load metadata from JSON file"""
        if os.path.exists(self.metadataFilepath):
            with open(self.metadataFilepath, 'r') as f:
                data = json.load(f)
                return ProgressMetadata.fromDict(data)
        return None


class NumpyProgressStorage(ProgressDataManager):
    """NumPy-based progress storage for binary format"""
    
    def __init__(self, filepath: str, metadata: Optional[ProgressMetadata] = None):
        if not filepath.endswith('.npz'):
            # If filepath is a directory or doesn't have extension, add .npz
            if os.path.isdir(filepath) or not os.path.splitext(filepath)[1]:
                filepath = os.path.join(filepath, 'progressAll.npz') if os.path.isdir(filepath) else filepath + '.npz'
            else:
                filepath = filepath.replace('.txt', '.npz')
        super().__init__(filepath, metadata)
        # Generate progressBest.npz instead of progressAll_best.npz for cleaner naming
        base_dir = os.path.dirname(filepath)
        self.bestFilepath = os.path.join(base_dir, 'progressBest.npz')
        self.metadataFilepath = filepath.replace('.npz', '_metadata.json')
    
    def saveEvaluation(self, record: EvaluationRecord, saveAll: bool = True) -> None:
        """Save evaluation record to NumPy files"""
        if saveAll:
            self.allRecords.append(record)
            self._saveRecordsToNpz(self.allRecords, self.filepath)
        
        if record.isBest:
            self.bestRecords.append(record)
            self._saveRecordsToNpz(self.bestRecords, self.bestFilepath)
    
    def _saveRecordsToNpz(self, records: List[EvaluationRecord], filepath: str) -> None:
        """Save records to NumPy compressed archive"""
        if not records:
            return
        
        evalNumbers = np.array([r.evaluationNumber for r in records])
        paramValues = np.array([r.parameterValues for r in records])
        elapsedTimes = np.array([r.elapsedTime for r in records])
        objectiveValues = np.array([r.objectiveValue for r in records])
        
        np.savez_compressed(
            filepath,
            evaluationNumbers=evalNumbers,
            parameterValues=paramValues,
            elapsedTimes=elapsedTimes,
            objectiveValues=objectiveValues
        )
    
    def loadData(self) -> Tuple[List[EvaluationRecord], List[EvaluationRecord], ProgressMetadata]:
        """Load data from NumPy files"""
        # Load metadata
        metadata = self._loadMetadata()
        
        # Load all records
        allRecords = []
        if os.path.exists(self.filepath):
            allRecords = self._loadRecordsFromNpz(self.filepath)
        
        # Load best records
        bestRecords = []
        if os.path.exists(self.bestFilepath):
            bestRecords = self._loadRecordsFromNpz(self.bestFilepath, isBest=True)
        
        self.allRecords = allRecords
        self.bestRecords = bestRecords
        self.metadata = metadata
        
        return bestRecords, allRecords, metadata
    
    def _loadRecordsFromNpz(self, filepath: str, isBest: bool = False) -> List[EvaluationRecord]:
        """Load records from NumPy compressed archive"""
        records = []
        
        with np.load(filepath) as data:
            evalNumbers = data['evaluationNumbers']
            paramValues = data['parameterValues']
            elapsedTimes = data['elapsedTimes']
            objectiveValues = data['objectiveValues']
            
            for i in range(len(evalNumbers)):
                record = EvaluationRecord(
                    evaluationNumber=int(evalNumbers[i]),
                    parameterValues=paramValues[i].tolist(),
                    elapsedTime=float(elapsedTimes[i]),
                    objectiveValue=float(objectiveValues[i]),
                    isBest=isBest
                )
                records.append(record)
        
        return records
    
    def saveMetadata(self) -> None:
        """Save metadata to JSON file"""
        if self.metadata:
            with open(self.metadataFilepath, 'w') as f:
                json.dump(self.metadata.toDict(), f, indent=2)
    
    def _loadMetadata(self) -> Optional[ProgressMetadata]:
        """Load metadata from JSON file"""
        if os.path.exists(self.metadataFilepath):
            with open(self.metadataFilepath, 'r') as f:
                data = json.load(f)
                return ProgressMetadata.fromDict(data)
        return None


def createProgressManager(
    filepath: str, 
    storageFormat: str = "csv", 
    metadata: Optional[ProgressMetadata] = None
) -> ProgressDataManager:
    """Factory function to create progress data manager"""
    if storageFormat.lower() == "csv":
        return CSVProgressStorage(filepath, metadata)
    elif storageFormat.lower() == "numpy":
        return NumpyProgressStorage(filepath, metadata)
    else:
        raise ValueError(f"Unsupported storage format: {storageFormat}")


def migrateLegacyProgressFile(
    legacyFilepath: str, 
    newFilepath: str,
    storageFormat: str = "csv",
    metadata: Optional[ProgressMetadata] = None
) -> None:
    """Migrate legacy progress.txt file to new format"""
    if not os.path.exists(legacyFilepath):
        return
    
    # Parse legacy file using existing parser
    import ast
    
    records = []
    with open(legacyFilepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                dataList = ast.literal_eval(line)
                if isinstance(dataList, list):
                    dataRow = [float(x) for x in dataList]
                    
                    # Assume structure: [evalNum, param1, param2, ..., elapsedTime, objectiveValue]
                    # or: [param1, param2, ..., elapsedTime, objectiveValue]
                    if len(dataRow) >= 3:
                        # Check if first value looks like evaluation number
                        if dataRow[0] == int(dataRow[0]) and dataRow[0] > 0:
                            evalNum = int(dataRow[0])
                            paramValues = dataRow[1:-2]
                            elapsedTime = dataRow[-2]
                            objectiveValue = dataRow[-1]
                        else:
                            evalNum = i + 1
                            paramValues = dataRow[:-2]
                            elapsedTime = dataRow[-2]
                            objectiveValue = dataRow[-1]
                        
                        record = EvaluationRecord(
                            evaluationNumber=evalNum,
                            parameterValues=paramValues,
                            elapsedTime=elapsedTime,
                            objectiveValue=objectiveValue,
                            isBest="progress.txt" in legacyFilepath  # Best if from progress.txt
                        )
                        records.append(record)
            except (ValueError, SyntaxError):
                continue
    
    # Save to new format
    manager = createProgressManager(newFilepath, storageFormat, metadata)
    manager.saveMetadata()
    
    for record in records:
        manager.saveEvaluation(record, saveAll=True)
    
    print(f"Migrated {len(records)} records from {legacyFilepath} to {newFilepath}")
