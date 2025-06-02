import os
import json
import datetime
from viennaps2d import Reader
from viennals2d import Reader as lsReader
from viennals2d import Domain as lsDomain


class Project:
    def __init__(self, name: str = None, projectDir: str = "./"):
        """
        Initialize a Fit project.

        Args:
            name: Name of the project. If provided, project paths will be set immediately.
            projectDir: Directory where the project will be created. Defaults to current directory.
        """
        # Set project name and paths
        self.projectName = name
        self.mode = "2D"  # Default mode

        # Domain related attributes
        self.initialDomain = None
        self.initialDomainPath = None
        self.targetLevelSet = None
        self.targetLevelSetPath = None

        # Set paths if name is provided
        if name is not None:
            base_name = name
            abs_base = os.path.abspath(os.path.join(projectDir, base_name))

            # Check if the project directory already exists
            index = 1
            while os.path.exists(abs_base):
                name = f"{base_name}_{index}"
                abs_base = os.path.abspath(os.path.join(projectDir, name))
                index += 1
                print(f"Project directory exists. Trying '{name}'...")

            self.projectName = name
            self.projectPath = os.path.join(projectDir, name)
            self.projectAbsPath = os.path.abspath(self.projectPath)
            self.projectInfoPath = os.path.join(
                self.projectAbsPath, f"{name}-info.json"
            )
            print(f"Project '{name}' will be created in '{self.projectAbsPath}'.")
        else:
            self.projectPath = None
            self.projectAbsPath = None
            self.projectInfoPath = None

    def initialize(self):
        """Initialize the project with a standard directory structure."""
        if self.projectName is None:
            raise ValueError("Project name must be provided during initialization.")

        # # Check if the project directory already exists
        # if os.path.exists(self.projectPath):
        #     raise FileExistsError(
        #         "Project already exists. Please choose a different name."
        #     )

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
        return self

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

        # If initialDomainPath is None, skip loading the initial domain
        if self.initialDomainPath == "":
            self.initialDomain = None
        else:
            # Load the initial domain
            self.initialDomain = Reader(
                os.path.join(
                    self.projectPath,
                    "domains",
                    "initialDomain",
                    f"{self.projectName}-initialDomain.vpsd",
                )
            ).apply()

        # If targetLevelSetPath is None, skip loading the target level set
        if self.targetLevelSetPath == "":
            self.targetLevelSet = None
        else:
            # Load the target level set
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

        # Create directories if needed
        initialDomainDir = os.path.join(self.projectPath, "domains", "initialDomain")
        os.makedirs(initialDomainDir, exist_ok=True)

        # Save the initial domain to the initial domain directory
        domainPath = os.path.join(
            initialDomainDir, f"{self.projectName}-initialDomain.vpsd"
        )

        if self.mode == "2D":
            import viennaps2d as vps
            import viennals2d as vls

            # Save the domain
            vps.Writer(domain, domainPath).apply()

            # Extract and visualize only the last level set (index -1)
            if domain.getLevelSets():
                lastLevelSet = domain.getLevelSets()[-1]

                # Create mesh visualization
                meshLS = vls.Mesh()
                vls.ToMesh(lastLevelSet, meshLS).apply()
                meshPath = os.path.join(
                    initialDomainDir, f"{self.projectName}-initialDomain-ls.vtp"
                )
                vls.VTKWriter(meshLS, meshPath).apply()

                # Create surface mesh visualization
                meshSurface = vls.Mesh()
                vls.ToSurfaceMesh(lastLevelSet, meshSurface).apply()
                surfacePath = os.path.join(
                    initialDomainDir, f"{self.projectName}-initialDomain-surface.vtp"
                )
                vls.VTKWriter(meshSurface, surfacePath).apply()

        elif self.mode == "3D":
            import viennaps3d as vps
            import viennals3d as vls

            vps.Writer(domain, domainPath).apply()

            # Extract and visualize only the last level set (index -1)
            if domain.getLevelSets():
                lastLevelSet = domain.getLevelSets()[-1]

                # Create mesh visualization
                meshLS = vls.Mesh()
                vls.ToMesh(lastLevelSet, meshLS).apply()
                meshPath = os.path.join(
                    initialDomainDir, f"{self.projectName}-initialDomain-ls.vtp"
                )
                vls.VTKWriter(meshLS, meshPath).apply()

                # Create surface mesh visualization
                meshSurface = vls.Mesh()
                vls.ToSurfaceMesh(lastLevelSet, meshSurface).apply()
                surfacePath = os.path.join(
                    initialDomainDir, f"{self.projectName}-initialDomain-surface.vtp"
                )
                vls.VTKWriter(meshSurface, surfacePath).apply()
        else:
            raise ValueError("Invalid mode. Only '2D' and '3D' are supported.")

        self.initialDomainPath = domainPath
        # Update project info with the initial domain path
        self.updateProjectInfo("initialDomainPath", domainPath)
        print(f"Initial domain and visualization meshes saved to {initialDomainDir}")
        return self

    def setTargetLevelSet(self, levelSet):
        """Set the target level set for the project."""
        self.targetLevelSet = levelSet

        # Create directories if needed
        targetDomainDir = os.path.join(self.projectPath, "domains", "targetDomain")
        os.makedirs(targetDomainDir, exist_ok=True)

        # Save the target level set
        domainPath = os.path.join(
            targetDomainDir, f"{self.projectName}-targetDomain.lvst"
        )

        if self.mode == "2D":
            import viennals2d as vls

            # Save the level set
            vls.Writer(levelSet, domainPath).apply()

            # Create mesh visualization
            meshLS = vls.Mesh()
            vls.ToMesh(levelSet, meshLS).apply()
            meshPath = os.path.join(
                targetDomainDir, f"{self.projectName}-targetDomain-ls.vtp"
            )
            vls.VTKWriter(meshLS, meshPath).apply()

            # Create surface mesh visualization
            meshSurface = vls.Mesh()
            vls.ToSurfaceMesh(levelSet, meshSurface).apply()
            surfacePath = os.path.join(
                targetDomainDir, f"{self.projectName}-targetDomain-surface.vtp"
            )
            vls.VTKWriter(meshSurface, surfacePath).apply()

        elif self.mode == "3D":
            import viennals3d as vls

            # Save the level set
            vls.Writer(levelSet, domainPath).apply()

            # Create mesh visualization
            meshLS = vls.Mesh()
            vls.ToMesh(levelSet, meshLS).apply()
            meshPath = os.path.join(
                targetDomainDir, f"{self.projectName}-targetDomain-ls.vtp"
            )
            vls.VTKWriter(meshLS, meshPath).apply()

            # Create surface mesh visualization
            meshSurface = vls.Mesh()
            vls.ToSurfaceMesh(levelSet, meshSurface).apply()
            surfacePath = os.path.join(
                targetDomainDir, f"{self.projectName}-targetDomain-surface.vtp"
            )
            vls.VTKWriter(meshSurface, surfacePath).apply()
        else:
            raise ValueError("Invalid mode. Only '2D' and '3D' are supported.")

        self.targetLevelSetPath = domainPath
        # Update project info with the target level set path
        self.updateProjectInfo("targetLevelSetPath", domainPath)
        print(f"Target level set and visualization meshes saved to {targetDomainDir}")
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

    def updateProjectInfo(self, field: str, value):
        """
        Update a field in the project info file.

        Args:
            field: Name of the field to update
            value: New value for the field

        Returns:
            self: For method chaining
        """
        if self.projectInfoPath is None or not os.path.exists(self.projectInfoPath):
            raise ValueError(
                "Project information file does not exist. Initialize or load a project first."
            )

        # Load current project info
        with open(self.projectInfoPath, "r") as f:
            projectInfo = json.load(f)

        # Update the specified field
        if field in projectInfo:
            projectInfo[field] = value
        else:
            print(
                f"Warning: Field '{field}' does not exist in project info. Adding it."
            )
            projectInfo[field] = value

        # Always update last modified date
        projectInfo["lastModifiedDate"] = str(datetime.datetime.now())

        # Save updated info back to file
        with open(self.projectInfoPath, "w") as f:
            json.dump(projectInfo, f, indent=4)

        # Also update the corresponding instance variable if it exists
        if hasattr(self, field):
            setattr(self, field, value)

        print(f"Updated project info field '{field}'")
        return self
