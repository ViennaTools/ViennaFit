import os
import json
import datetime
from viennaps2d import Reader
from viennals2d import Reader as lsReader
from viennals2d import Domain as lsDomain


class Project:
    def __init__(self):
        self.projectName = None
        self.projectPath = None
        self.projectAbsPath = None
        self.projectInfoPath = None

        self.mode = "2D"  # Default mode

        self.initialDomain = None
        self.initialDomainPath = None
        self.targetLevelSet = None
        self.targetLevelSetPath = None

    def setName(self, name: str):
        """Set the name of the project."""
        self.projectName = name
        self.projectPath = f"./{name}"
        self.projectAbsPath = os.path.abspath(self.projectPath)
        self.projectInfoPath = os.path.join(
            self.projectAbsPath, f"{self.projectName}-info.json"
        )
        print(f"Project name set to '{self.projectName}'.")

    def initialize(self):
        """Initialize the project with a standard directory structure."""
        if self.projectName is None:
            raise ValueError("Project name must be set before initialization.")

        # Check if the project directory already exists
        if os.path.exists(self.projectPath):
            raise FileExistsError(
                "Project already exists. Please choose a different name."
            )

        # Create the main project directory
        os.makedirs(self.projectPath, exist_ok=True)

        # Create subdirectories
        subdirs = [
            "domains",
        ]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.projectPath, subdir), exist_ok=True)

        # Create folders within the domains directory
        domainSubdirs = [
            "annotations",
            "initialDomain",
            "targetDomain",
            "optimalDomains",
        ]
        for subdir in domainSubdirs:
            os.makedirs(
                os.path.join(self.projectPath, "domains", subdir), exist_ok=True
            )

        print(f"Project '{self.projectName}' initialized with standard structure.")

        # Create project information dictionary
        projectInfo = {
            "projectName": self.projectName,
            "projectDescription": "This is a sample project description.",
            "projectPath": self.projectAbsPath,
            "createdDate": str(datetime.datetime.now()),
            "lastModifiedDate": str(datetime.datetime.now()),
            "mode": self.mode,
            "initialDomainPath": "",
            "targetLevelSetPath": "",
        }

        # Save project information to JSON file
        with open(self.projectInfoPath, "w") as f:
            json.dump(projectInfo, f, indent=4)

        print(f"Project information saved to {self.projectInfoPath}")

    def load(self, projectPath: str = "projectPath"):
        if not os.path.exists(projectPath):
            raise FileNotFoundError(
                f"Project directory '{projectPath}' does not exist."
            )

        # if projethPath does not end with .json, construct the path
        if not projectPath.endswith(".json"):
            # project name is last part of the path
            projectName = os.path.basename(projectPath)
            print(projectName)
            projectInfoPath = os.path.join(projectPath, f"{projectName}-info.json")

        # load the project information from the JSON file
        with open(projectInfoPath, "r") as f:
            projectInfo = json.load(f)
        self.projectName = projectInfo["projectName"]
        self.projectPath = projectInfo["projectPath"]
        self.projectAbsPath = os.path.abspath(self.projectPath)
        self.projectInfoPath = projectInfoPath
        self.mode = projectInfo["mode"]
        self.initialDomainPath = projectInfo["initialDomainPath"]
        self.targetLevelSetPath = projectInfo["targetLevelSetPath"]
        self.optimizationRuns = projectInfo["optimizationRuns"]
        self.locSensStudies = projectInfo["locSensStudies"]
        self.globSensStudies = projectInfo["globSensStudies"]

        self.initialDomain = Reader(
            os.path.join(
                self.projectPath,
                "domains",
                "initialDomain",
                f"{self.projectName}-initialDomain.vpsd",
            )
        ).apply()

        targetDomain = lsDomain()
        lsReader(
            targetDomain,
            os.path.join(
                self.projectPath,
                "domains",
                "targetDomain",
                f"{self.projectName}-targetDomain.lvst",
            ),
        ).apply()
        self.targetLevelSet = targetDomain

        # Print project information
        print(f"Project '{self.projectName}' loaded with the following information:")
        print(json.dumps(projectInfo, indent=4))

    def setMode(self, mode: str):
        """Set the mode of the project. Currently only 2D is supported."""
        if mode not in ["2D", "3D"]:
            raise ValueError("Invalid mode. Only '2D' and '3D' are supported.")
        self.mode = mode
        print(f"Project mode set to {self.mode}.")

    def setInitialDomain(self, domain):
        """Set the initial domain for the project."""
        self.initialDomain = domain

        # Save the initial domain to the inital domain directory
        initialDomainDir = os.path.join(self.projectPath, "domains", "initialDomain")
        os.makedirs(initialDomainDir, exist_ok=True)
        domainPath = os.path.join(
            initialDomainDir, f"{self.projectName}-initialDomain.vpsd"
        )

        if self.mode == "2D":
            import viennaps2d as vps

            vps.Writer(domain, domainPath).apply()
        elif self.mode == "3D":
            import viennaps3d as vps

            vps.Writer(domain, domainPath).apply()
        else:
            raise ValueError("Invalid mode. Only '2D' and '3D' are supported.")
        self.initialDomainPath = domainPath
        print(f"Initial domain saved to {initialDomainDir}")
        return self

    def setTargetLevelSet(self, levelSet):
        """Set the target level set for the project."""
        self.targetLevelSet = levelSet

        # Save the target level set to the target domain directory
        targetDomainDir = os.path.join(self.projectPath, "domains", "targetDomain")
        os.makedirs(targetDomainDir, exist_ok=True)
        domainPath = os.path.join(
            targetDomainDir, f"{self.projectName}-targetDomain.lvst"
        )

        if self.mode == "2D":
            import viennals2d as vls

            vls.Writer(levelSet, domainPath).apply()
        elif self.mode == "3D":
            import viennals3d as vls

            vls.Writer(levelSet, domainPath).apply()
        else:
            raise ValueError("Invalid mode. Only '2D' and '3D' are supported.")

        self.targetLevelSetPath = domainPath
        print(f"Target level set saved to {targetDomainDir}")
        return self

    def isReady(self):
        """Check if the project is ready for optimization/sensitivity studies."""
        if self.initialDomain is None:
            print("Initial domain is not set.")
            return False
        if self.targetLevelSet is None:
            print("Target level set is not set.")
            return False
        return True
