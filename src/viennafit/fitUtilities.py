import json
import numpy as np
import time


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
    # limit the number of decimal places to 5
    evaluation = [round(i, 5) for i in evaluation]
    evaluation = [str(i) for i in evaluation]

    with open(filepath, "a") as file:
        file.write(str(evaluation) + "\n")


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
