import json
import numpy as np
import time
import os
import csv
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata


def getViennaVersionInfo() -> Dict[str, Optional[str]]:
    """
    Get version information for viennaps and viennals packages.

    Attempts to retrieve:
    - Package versions via importlib.metadata or __version__ attribute
    - Git commit hashes if packages were installed from local git repositories

    Returns:
        Dictionary with keys:
        - viennapsVersion: version string or "unknown"
        - viennalsVersion: version string or "unknown"
        - viennapsCommit: git commit hash or None
        - viennalsCommit: git commit hash or None
    """
    import subprocess
    from urllib.parse import urlparse

    result = {
        "viennapsVersion": "unknown",
        "viennalsVersion": "unknown",
        "viennapsCommit": None,
        "viennalsCommit": None,
    }

    # Get viennaps version
    try:
        result["viennapsVersion"] = importlib_metadata.version("viennaps")
    except importlib_metadata.PackageNotFoundError:
        try:
            import viennaps as vps
            result["viennapsVersion"] = getattr(vps, "__version__", "unknown")
        except ImportError:
            pass

    # Get viennals version
    try:
        result["viennalsVersion"] = importlib_metadata.version("viennals")
    except importlib_metadata.PackageNotFoundError:
        try:
            import viennals as vls
            result["viennalsVersion"] = getattr(vls, "__version__", "unknown")
        except ImportError:
            pass

    # Try to get git commit hashes from source directories
    for pkg_name, commit_key in [("viennaps", "viennapsCommit"), ("viennals", "viennalsCommit")]:
        try:
            dist = importlib_metadata.distribution(pkg_name)
            # Check for direct_url.json which contains source directory info
            direct_url_files = [f for f in dist.files or [] if f.name == "direct_url.json"]
            if direct_url_files:
                direct_url_path = direct_url_files[0].locate()
                if os.path.exists(direct_url_path):
                    with open(direct_url_path, "r") as f:
                        direct_url_data = json.load(f)

                        # First check for vcs_info (VCS URL installs)
                        vcs_info = direct_url_data.get("vcs_info", {})
                        commit_id = vcs_info.get("commit_id")
                        if commit_id:
                            result[commit_key] = commit_id
                            continue

                        # Check for local file URL (pip install from local directory)
                        url = direct_url_data.get("url", "")
                        if url.startswith("file://"):
                            parsed = urlparse(url)
                            source_dir = parsed.path
                            # Check if this directory exists and is a git repo
                            if os.path.isdir(source_dir) and os.path.isdir(os.path.join(source_dir, ".git")):
                                try:
                                    git_result = subprocess.run(
                                        ["git", "rev-parse", "HEAD"],
                                        cwd=source_dir,
                                        capture_output=True,
                                        text=True,
                                        timeout=5
                                    )
                                    if git_result.returncode == 0:
                                        result[commit_key] = git_result.stdout.strip()
                                except (subprocess.SubprocessError, OSError):
                                    pass
        except (importlib_metadata.PackageNotFoundError, Exception):
            pass

    return result


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
    saveAll: bool = True,
    simulationTime: float = 0.0,
    distanceMetricTime: float = 0.0,
    totalElapsedTime: float = 0.0,
    additionalMetricValues: Optional[Dict[str, float]] = None,
    additionalMetricTimes: Optional[Dict[str, float]] = None
) -> None:
    """Save evaluation using the new progress manager system"""
    record = EvaluationRecord(
        evaluationNumber=evaluationNumber,
        parameterValues=parameterValues,
        elapsedTime=elapsedTime,
        objectiveValue=objectiveValue,
        isBest=isBest,
        simulationTime=simulationTime,
        distanceMetricTime=distanceMetricTime,
        totalElapsedTime=totalElapsedTime,
        additionalMetricValues=additionalMetricValues,
        additionalMetricTimes=additionalMetricTimes
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
    numEvaluations: Optional[int] = None
    notes: Optional[str] = None
    viennapsVersion: Optional[str] = None
    viennalsVersion: Optional[str] = None
    viennapsCommit: Optional[str] = None
    viennalsCommit: Optional[str] = None

    def toDict(self) -> Dict[str, Any]:
        result = {
            "runName": self.runName,
            "parameterNames": self.parameterNames,
            "parameterBounds": self.parameterBounds,
            "fixedParameters": self.fixedParameters,
            "optimizer": self.optimizer,
            "createdTime": self.createdTime,
            "description": self.description
        }
        # Include numEvaluations and notes if available
        if self.numEvaluations is not None:
            result["numEvaluations"] = self.numEvaluations
        if self.notes is not None:
            result["notes"] = self.notes
        # Include version info if available
        if self.viennapsVersion is not None:
            result["viennapsVersion"] = self.viennapsVersion
        if self.viennalsVersion is not None:
            result["viennalsVersion"] = self.viennalsVersion
        if self.viennapsCommit is not None:
            result["viennapsCommit"] = self.viennapsCommit
        if self.viennalsCommit is not None:
            result["viennalsCommit"] = self.viennalsCommit
        return result

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> 'ProgressMetadata':
        # Handle backward compatibility - filter to only known fields
        known_fields = {
            "runName", "parameterNames", "parameterBounds", "fixedParameters",
            "optimizer", "createdTime", "description", "numEvaluations", "notes",
            "viennapsVersion", "viennalsVersion", "viennapsCommit", "viennalsCommit"
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


@dataclass
class EvaluationRecord:
    """Single evaluation record"""
    evaluationNumber: int
    parameterValues: List[float]
    elapsedTime: float
    objectiveValue: float
    isBest: bool = False
    simulationTime: float = 0.0
    distanceMetricTime: float = 0.0
    totalElapsedTime: float = 0.0  # Cumulative time since optimization started
    additionalMetricValues: Optional[Dict[str, float]] = None
    additionalMetricTimes: Optional[Dict[str, float]] = None

    def toDict(self) -> Dict[str, Any]:
        result = {
            "evaluationNumber": self.evaluationNumber,
            "parameterValues": self.parameterValues,
            "elapsedTime": self.elapsedTime,
            "objectiveValue": self.objectiveValue,
            "isBest": self.isBest,
            "simulationTime": self.simulationTime,
            "distanceMetricTime": self.distanceMetricTime,
            "totalElapsedTime": self.totalElapsedTime
        }
        if self.additionalMetricValues:
            result["additionalMetricValues"] = self.additionalMetricValues
        if self.additionalMetricTimes:
            result["additionalMetricTimes"] = self.additionalMetricTimes
        return result


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
                fieldnames = ['evaluationNumber'] + self.metadata.parameterNames + ['elapsedTime', 'simulationTime', 'distanceMetricTime', 'totalElapsedTime', 'objectiveValue']
            else:
                fieldnames = ['evaluationNumber'] + [f'param_{i}' for i in range(actualParamCount)] + ['elapsedTime', 'simulationTime', 'distanceMetricTime', 'totalElapsedTime', 'objectiveValue']

            # Add additional metric columns if present
            if record.additionalMetricValues:
                for metricName in sorted(record.additionalMetricValues.keys()):
                    fieldnames.append(f'{metricName}_value')
                    if record.additionalMetricTimes and metricName in record.additionalMetricTimes:
                        fieldnames.append(f'{metricName}_time')

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not fileExists:
                writer.writeheader()

            # Write data row
            row = {'evaluationNumber': record.evaluationNumber}
            paramFieldnames = fieldnames[1:1+actualParamCount]  # Parameter columns
            for i, paramName in enumerate(paramFieldnames):
                if i < len(record.parameterValues):
                    row[paramName] = record.parameterValues[i]
                else:
                    row[paramName] = 0.0  # Default value if not enough parameters
            row['elapsedTime'] = record.elapsedTime
            row['simulationTime'] = record.simulationTime
            row['distanceMetricTime'] = record.distanceMetricTime
            row['totalElapsedTime'] = record.totalElapsedTime
            row['objectiveValue'] = record.objectiveValue

            # Add additional metric values and times
            if record.additionalMetricValues:
                for metricName in sorted(record.additionalMetricValues.keys()):
                    row[f'{metricName}_value'] = record.additionalMetricValues[metricName]
                    if record.additionalMetricTimes and metricName in record.additionalMetricTimes:
                        row[f'{metricName}_time'] = record.additionalMetricTimes[metricName]

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

                # Read new timing fields if available (backward compatibility)
                simulationTime = float(row.get('simulationTime', 0.0))
                distanceMetricTime = float(row.get('distanceMetricTime', 0.0))
                totalElapsedTime = float(row.get('totalElapsedTime', 0.0))

                # Extract parameter values and additional metrics
                paramValues = []
                additionalMetricValues = {}
                additionalMetricTimes = {}

                excludeKeys = ['evaluationNumber', 'elapsedTime', 'simulationTime', 'distanceMetricTime', 'totalElapsedTime', 'objectiveValue']

                for key, value in row.items():
                    if key not in excludeKeys:
                        if key.endswith('_value'):
                            # Additional metric value
                            metricName = key[:-6]  # Remove '_value' suffix
                            additionalMetricValues[metricName] = float(value)
                        elif key.endswith('_time'):
                            # Additional metric timing
                            metricName = key[:-5]  # Remove '_time' suffix
                            additionalMetricTimes[metricName] = float(value)
                        else:
                            # Regular parameter value
                            paramValues.append(float(value))

                record = EvaluationRecord(
                    evaluationNumber=evalNum,
                    parameterValues=paramValues,
                    elapsedTime=elapsedTime,
                    objectiveValue=objectiveValue,
                    isBest=(filepath == self.bestFilepath),
                    simulationTime=simulationTime,
                    distanceMetricTime=distanceMetricTime,
                    totalElapsedTime=totalElapsedTime,
                    additionalMetricValues=additionalMetricValues if additionalMetricValues else None,
                    additionalMetricTimes=additionalMetricTimes if additionalMetricTimes else None
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
        simulationTimes = np.array([r.simulationTime for r in records])
        distanceMetricTimes = np.array([r.distanceMetricTime for r in records])
        objectiveValues = np.array([r.objectiveValue for r in records])

        np.savez_compressed(
            filepath,
            evaluationNumbers=evalNumbers,
            parameterValues=paramValues,
            elapsedTimes=elapsedTimes,
            simulationTimes=simulationTimes,
            distanceMetricTimes=distanceMetricTimes,
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

            # Load new timing fields if available (backward compatibility)
            simulationTimes = data.get('simulationTimes', np.zeros(len(evalNumbers)))
            distanceMetricTimes = data.get('distanceMetricTimes', np.zeros(len(evalNumbers)))

            for i in range(len(evalNumbers)):
                record = EvaluationRecord(
                    evaluationNumber=int(evalNumbers[i]),
                    parameterValues=paramValues[i].tolist(),
                    elapsedTime=float(elapsedTimes[i]),
                    objectiveValue=float(objectiveValues[i]),
                    isBest=isBest,
                    simulationTime=float(simulationTimes[i]),
                    distanceMetricTime=float(distanceMetricTimes[i])
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


def plotParameterProgression(
    fileName,
    plotTitle="",
    logscale=None,
    logscaleObjective=False,
    logscaleParameters=False,
    numberOfEvaluations=0,
    fontSize=1.0,
    dpi=300,
    allXTicks=False,
    skipJumps=False,
    skipParameters=None,
):
    """
    Plot parameter progression showing how parameters evolve during optimization.
    Similar to plotProgressFileH1 but focuses on parameter values vs evaluation number.

    Args:
        fileName: Path to progress file (CSV or legacy TXT format)
        plotTitle: Title for the plot
        logscale: Legacy parameter - applies log scale to both objective and parameters (deprecated, use logscaleObjective/logscaleParameters)
        logscaleObjective: Whether to use logarithmic scale for objective function y-axis
        logscaleParameters: Whether to use logarithmic scale for parameter y-axes
        numberOfEvaluations: Limit number of evaluations to plot (0 = all)
        fontSize: Font size multiplier
        dpi: Plot DPI
        allXTicks: Whether to show all x-tick values
        skipJumps: Whether to skip evaluations that don't improve objective
        skipParameters: List of parameter names to exclude from plotting (e.g., ['param1', 'param2'])

    Returns:
        Path to created plot file
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import ast

    # Handle backward compatibility for legacy logscale parameter
    if logscale is not None:
        logscaleObjective = logscale
        logscaleParameters = logscale

    # Initialize skipParameters as empty list if None
    if skipParameters is None:
        skipParameters = []

    # Set font to Times and apply font size multiplier
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    originalFontSize = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = originalFontSize * fontSize

    try:
        # Read and parse the progress file
        progressData, paramNames = _readProgressFile(fileName)
        if progressData is None or len(paramNames) == 0:
            raise ValueError(f"Could not read progress data from {fileName}")

        # Filter out skipped parameters
        if skipParameters:
            # Find indices of parameters to keep
            keepIndices = []
            filteredParamNames = []
            for i, paramName in enumerate(paramNames):
                if paramName not in skipParameters:
                    keepIndices.append(i)
                    filteredParamNames.append(paramName)

            # Update parameter names
            paramNames = filteredParamNames

            # Filter the progress data to keep only non-skipped parameters
            if keepIndices:
                # Keep selected parameter columns + objective (last column)
                filteredProgressData = progressData[:, keepIndices + [-1]]
                progressData = filteredProgressData
            else:
                # If all parameters are skipped, only keep objective
                progressData = progressData[:, [-1]]
                paramNames = []

        # Apply evaluation limit
        if numberOfEvaluations > 0:
            progressData = progressData[:numberOfEvaluations]
        else:
            numberOfEvaluations = len(progressData)

        # Filter data if skipJumps is enabled
        if skipJumps:
            filteredData = []
            filteredIndices = []
            lastValue = float("inf")

            for i, row in enumerate(progressData):
                currentValue = row[-1]  # Objective value is last column
                if currentValue < lastValue:
                    filteredData.append(row)
                    filteredIndices.append(i + 1)  # +1 because plot indices start from 1
                    lastValue = currentValue

            progressData = np.array(filteredData) if filteredData else progressData
            indexToUse = filteredIndices
        else:
            indexToUse = list(range(1, len(progressData) + 1))

        # Find minimum objective for reporting
        minIndex = np.argmin(progressData[:, -1])
        minRow = progressData[minIndex]
        print("minIndex: ", minIndex)
        print("minRow: ", minRow)

        # Create subplots for each parameter + objective function
        numParams = len(paramNames)
        numSubplots = numParams + 1  # +1 for objective function

        fig, axs = plt.subplots(
            numSubplots, 1,
            figsize=(3.5, max(7.5, numSubplots * 1.2)),
            dpi=dpi
        )

        # Ensure axs is always a list
        if numSubplots == 1:
            axs = [axs]

        # Plot objective function (last column)
        objValues = [float(row[-1]) for row in progressData]
        axs[0].plot(indexToUse, objValues, linewidth=1.5)
        axs[0].scatter(indexToUse, objValues, color="red", s=15)
        axs[0].set_ylabel("Objective func.", fontsize=plt.rcParams["font.size"])

        # Apply log scale to objective function if requested
        if logscaleObjective:
            axs[0].set_yscale("log")

        # Plot each parameter
        colors = ['blue', 'green', 'orange', 'cyan', 'brown', 'purple', 'pink', 'gray']
        for i, paramName in enumerate(paramNames):
            paramValues = [float(row[i]) for row in progressData]
            ax = axs[i + 1]
            color = colors[i % len(colors)]

            ax.plot(indexToUse, paramValues, linewidth=1.5, color=color)
            ax.scatter(indexToUse, paramValues, color=color, s=15)
            ax.set_ylabel(paramName, fontsize=plt.rcParams["font.size"])

            # Apply log scale to parameter if requested
            if logscaleParameters:
                ax.set_yscale("log")

        # Set x-label on bottom subplot
        axs[-1].set_xlabel("Evaluation number", fontsize=plt.rcParams["font.size"])

        # Configure all axes
        for i, ax in enumerate(axs):
            ax.tick_params(axis="both", labelsize=plt.rcParams["font.size"] * 0.8)
            ax.locator_params(axis="x", nbins=5)

            # Only set nbins for y-axis if not using log scale
            # Check log scale setting for each subplot individually
            if i == 0:  # Objective function
                if not logscaleObjective:
                    ax.locator_params(axis="y", nbins=4)
            else:  # Parameter plots
                if not logscaleParameters:
                    ax.locator_params(axis="y", nbins=4)

            ax.grid(True, alpha=0.3)

        # Handle x-ticks
        if allXTicks:
            for ax in axs:
                ax.set_xticks(range(1, numberOfEvaluations + 1))

        # Adjust layout with compact spacing for IEEE column format
        plt.tight_layout(pad=0.8, h_pad=0.5)

        # Set title
        if plotTitle:
            fig.suptitle(plotTitle, fontsize=plt.rcParams["font.size"])

        # Generate output path with appropriate suffix
        basename = os.path.basename(fileName)

        # Create suffix based on log scale settings
        if logscaleObjective and logscaleParameters:
            suffix = "_log"  # Both (backward compatible)
        elif logscaleObjective:
            suffix = "_obj_log"  # Objective only
        elif logscaleParameters:
            suffix = "_params_log"  # Parameters only
        else:
            suffix = ""  # Neither

        writePath = fileName.split(".")[0] + f"_parameter_progression{suffix}.png"

        # Save plot
        plt.savefig(writePath, bbox_inches="tight", dpi=dpi)
        plt.close()

        print(f"Parameter progression plot saved to: {writePath}")
        return writePath

    finally:
        # Restore original font size
        plt.rcParams["font.size"] = originalFontSize


def _readProgressFile(fileName):
    """
    Read progress file in either CSV or legacy TXT format.

    Returns:
        tuple: (progressData as numpy array, parameterNames as list)
    """
    import pandas as pd
    import ast

    if not os.path.exists(fileName):
        raise FileNotFoundError(f"Progress file not found: {fileName}")

    # Try CSV format first
    if fileName.endswith('.csv'):
        try:
            df = pd.read_csv(fileName)
            # Extract parameter names (exclude non-parameter columns)
            excludeCols = ['evaluationNumber', 'elapsedTime', 'objectiveValue']
            paramNames = [col for col in df.columns if col not in excludeCols]

            # Construct data array: [param1, param2, ..., objectiveValue]
            paramData = df[paramNames].values if paramNames else np.array([])
            if 'objectiveValue' in df.columns:
                objData = df['objectiveValue'].values.reshape(-1, 1)
                if paramData.size > 0:
                    progressData = np.column_stack([paramData, objData])
                else:
                    progressData = objData
                    paramNames = []
            else:
                progressData = paramData

            return progressData, paramNames
        except Exception as e:
            print(f"Failed to read CSV format: {e}")

    # Try legacy TXT format
    try:
        with open(fileName, 'r') as f:
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

        if not dataRows:
            raise ValueError("No valid data found in file")

        # Convert to numpy array
        maxCols = max(len(row) for row in dataRows)
        paddedRows = []
        for row in dataRows:
            if len(row) < maxCols:
                row.extend([np.nan] * (maxCols - len(row)))
            paddedRows.append(row)

        progressData = np.array(paddedRows)

        # Assume format: [param1, param2, ..., elapsedTime, objectiveValue]
        # Create generic parameter names
        numParams = progressData.shape[1] - 2  # -2 for elapsedTime and objectiveValue
        paramNames = [f"param{i+1}" for i in range(max(0, numParams))]

        return progressData, paramNames

    except Exception as e:
        print(f"Failed to read TXT format: {e}")
        raise ValueError(f"Could not parse progress file {fileName}")


def plotParameterPositions(
    resultsFile=None,
    bestParameters=None,
    parameterBounds=None,
    plotTitle="Parameter Positions",
    skipParameters=None,
    fontSize=1.0,
    dpi=300,
    outputPath=None,
):
    """
    Plot optimal parameter positions within their bounds using a compact horizontal bar chart.

    Args:
        resultsFile: Path to results JSON file (alternative to providing bestParameters/bounds directly)
        bestParameters: Dictionary of parameter names to optimal values
        parameterBounds: Dictionary of parameter names to [min, max] bounds
        plotTitle: Title for the plot
        skipParameters: List of parameter names to exclude from plotting
        fontSize: Font size multiplier
        dpi: Plot DPI
        outputPath: Custom output path (if None, auto-generated from input)

    Returns:
        Path to created plot file

    Example:
        # From results file
        plotParameterPositions("optimization_results.json")

        # Direct input
        plotParameterPositions(
            bestParameters={'param1': 1.5, 'param2': 2.3},
            parameterBounds={'param1': [1.0, 2.0], 'param2': [2.0, 3.0]}
        )
    """
    import matplotlib.pyplot as plt
    import json

    # Initialize skipParameters as empty list if None
    if skipParameters is None:
        skipParameters = []

    # Load data from results file if provided
    if resultsFile is not None:
        if not os.path.exists(resultsFile):
            raise FileNotFoundError(f"Results file not found: {resultsFile}")

        with open(resultsFile, 'r') as f:
            results = json.load(f)

        if bestParameters is None:
            bestParameters = results.get('bestParameters', {})
        if parameterBounds is None:
            parameterBounds = results.get('variableParameters', results.get('parameterBounds', {}))

    # Validate inputs
    if not bestParameters:
        raise ValueError("No best parameters provided. Either specify resultsFile or bestParameters.")
    if not parameterBounds:
        raise ValueError("No parameter bounds provided. Either specify resultsFile or parameterBounds.")

    # Filter parameters based on skipParameters
    filteredParams = {
        name: value for name, value in bestParameters.items()
        if name not in skipParameters and name in parameterBounds
    }

    if not filteredParams:
        raise ValueError("No parameters to plot after filtering.")

    # Set font styling
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    originalFontSize = plt.rcParams["font.size"]
    plt.rcParams["font.size"] = originalFontSize * fontSize

    try:
        # Prepare data for plotting
        paramNames = list(filteredParams.keys())
        paramValues = list(filteredParams.values())
        bounds = [parameterBounds[name] for name in paramNames]

        # Calculate positions and percentages
        positions = []
        percentages = []
        rangeWidths = []

        for i, (name, value) in enumerate(filteredParams.items()):
            minBound, maxBound = parameterBounds[name]
            rangeWidth = maxBound - minBound
            rangeWidths.append(rangeWidth)

            # Calculate position as percentage of range
            if rangeWidth > 0:
                percentage = ((value - minBound) / rangeWidth) * 100
            else:
                percentage = 50  # Default to middle if no range
            percentages.append(percentage)
            positions.append(value)

        # Create horizontal bar chart with normalized x-axis
        numParams = len(paramNames)
        figHeight = max(3.5, numParams * 0.8 + 1.5)  # Slightly more space for annotations
        fig, ax = plt.subplots(figsize=(10, figHeight), dpi=dpi)  # Wider for bounds annotations

        # Create normalized horizontal bars (all parameters from 0 to 1)
        for i, (name, value) in enumerate(filteredParams.items()):
            minBound, maxBound = parameterBounds[name]
            y_pos = numParams - 1 - i  # Reverse order for top-to-bottom display

            # Calculate normalized position (0 to 1)
            if maxBound != minBound:
                normalizedValue = (value - minBound) / (maxBound - minBound)
            else:
                normalizedValue = 0.5  # Middle if no range

            # Draw full-width normalized range bar (0 to 1)
            ax.barh(
                y_pos,
                1.0,  # Full width = 100% of range
                left=0.0,
                height=0.6,
                color='lightblue',
                alpha=0.7,
                edgecolor='navy',
                linewidth=1
            )

            # Add optimal value marker at normalized position
            ax.scatter(
                normalizedValue, y_pos,
                color='red',
                s=80,
                zorder=5,
                marker='o',
                edgecolor='darkred',
                linewidth=1.5
            )

            # Add bounds annotations at bar edges
            # Left bound (minimum)
            ax.annotate(
                f'{minBound:.3g}',
                xy=(0.0, y_pos),
                xytext=(-5, 0),
                textcoords='offset points',
                va='center',
                ha='right',
                fontsize=plt.rcParams["font.size"] * 0.8,
                color='navy',
                fontweight='bold'
            )

            # Right bound (maximum)
            ax.annotate(
                f'{maxBound:.3g}',
                xy=(1.0, y_pos),
                xytext=(5, 0),
                textcoords='offset points',
                va='center',
                ha='left',
                fontsize=plt.rcParams["font.size"] * 0.8,
                color='navy',
                fontweight='bold'
            )

            # Add optimal value and percentage annotation above the marker
            percentage = normalizedValue * 100
            ax.annotate(
                f'{value:.3g} ({percentage:.1f}%)',
                xy=(normalizedValue, y_pos),
                xytext=(0, 15),
                textcoords='offset points',
                va='bottom',
                ha='center',
                fontsize=plt.rcParams["font.size"] * 0.85,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor='gray'
                )
            )

        # Format axes with normalized scale
        ax.set_xlim(-0.1, 1.1)  # Slight padding for bounds annotations
        ax.set_yticks(range(numParams))
        ax.set_yticklabels([paramNames[numParams - 1 - i] for i in range(numParams)])

        # Set up normalized x-axis
        ax.set_xlabel('Normalized Position within Parameter Bounds', fontsize=plt.rcParams["font.size"])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_title(plotTitle, fontsize=plt.rcParams["font.size"] * 1.1, pad=20)

        # Add grid for easier reading
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)

        # Add vertical reference lines at quartiles
        for pos in [0.25, 0.5, 0.75]:
            ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.3, zorder=1)

        # Adjust layout to accommodate external legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for external legend

        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legendElements = [
            Patch(facecolor='lightblue', alpha=0.7, edgecolor='navy', label='Normalized Parameter Range (0-100%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, markeredgecolor='darkred', label='Optimal Value Position'),
            Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='Reference Lines (25%, 50%, 75%)')
        ]

        ax.legend(
            handles=legendElements,
            bbox_to_anchor=(1.05, 1),  # Outside plot area, top-right
            loc='upper left',           # Anchor point on legend
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=plt.rcParams["font.size"] * 0.8
        )

        # Generate output path
        if outputPath is None:
            if resultsFile:
                baseName = os.path.splitext(resultsFile)[0]
                outputPath = f"{baseName}_parameter_positions.png"
            else:
                outputPath = "parameter_positions.png"

        # Save plot
        plt.savefig(outputPath, bbox_inches='tight', dpi=dpi)
        plt.close()

        print(f"Parameter positions plot saved to: {outputPath}")
        return outputPath

    finally:
        # Restore original font size
        plt.rcParams["font.size"] = originalFontSize
