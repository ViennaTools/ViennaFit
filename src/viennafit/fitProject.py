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
        
        # Multi-domain support (dictionary-based)
        self.initialDomains = {}  # {name: domain}
        self.initialDomainPaths = {}  # {name: path}
        
        # Multi-target support (dictionary-based)
        self.targetLevelSets = {}  # {name: levelSet}
        self.targetLevelSetPaths = {}  # {name: path}

        # Set paths if name is provided
        if name is not None:
            baseName = name
            absBase = os.path.abspath(os.path.join(projectDir, baseName))

            # Check if the project directory already exists
            if os.path.exists(absBase):
                # Find the highest existing index
                maxIndex = 0

                if os.path.exists(projectDir):
                    for existingDir in os.listdir(projectDir):
                        if (
                            existingDir.startswith(f"{baseName}_")
                            and existingDir[len(baseName) + 1 :].isdigit()
                        ):
                            existingIndex = int(existingDir[len(baseName) + 1 :])
                            maxIndex = max(maxIndex, existingIndex)

                # Use the next available index
                newIndex = maxIndex + 1
                name = f"{baseName}_{newIndex}"
                print(
                    f"Project directory already exists. Renaming project to '{name}'..."
                )

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
            "initialDomainPath": "",  # Keep for backward compatibility
            "initialDomainPaths": {},  # New multi-domain support
            "targetLevelSetPath": "",  # Keep for backward compatibility
            "targetLevelSetPaths": {},  # New multi-target support
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
        
        # Load multi-domain paths (new format) with backward compatibility
        if "initialDomainPaths" in projectInfo:
            self.initialDomainPaths = projectInfo["initialDomainPaths"]
        else:
            # Old format: create multi-domain structure from single domain
            self.initialDomainPaths = {}
            if self.initialDomainPath != "":
                self.initialDomainPaths["default"] = self.initialDomainPath
        
        # Load multi-target paths (new format) with backward compatibility
        if "targetLevelSetPaths" in projectInfo:
            self.targetLevelSetPaths = projectInfo["targetLevelSetPaths"]
        else:
            # Old format: create multi-target structure from single target
            self.targetLevelSetPaths = {}
            if self.targetLevelSetPath != "":
                self.targetLevelSetPaths["default"] = self.targetLevelSetPath

        # Load single initial domain (backward compatibility)
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

        # Load single target (backward compatibility)
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

        # Load all named targets (new multi-target support)
        self.targetLevelSets = {}
        for targetName, targetPath in self.targetLevelSetPaths.items():
            if targetPath != "" and os.path.exists(targetPath):
                targetDomain = lsDomain()
                lsReader(targetDomain, targetPath).apply()
                self.targetLevelSets[targetName] = targetDomain
                
                # If this is the default target and we don't have a single target loaded, use it
                if targetName == "default" and self.targetLevelSet is None:
                    self.targetLevelSet = targetDomain

        # Load all named initial domains (new multi-domain support)
        self.initialDomains = {}
        for domainName, domainPath in self.initialDomainPaths.items():
            if domainPath != "" and os.path.exists(domainPath):
                domain = Reader(domainPath).apply()
                self.initialDomains[domainName] = domain
                
                # If this is the default domain and we don't have a single domain loaded, use it
                if domainName == "default" and self.initialDomain is None:
                    self.initialDomain = domain

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
        """Set the initial domain for the project (backward compatibility method)."""
        # Maintain backward compatibility by using a default domain name
        defaultDomainName = "default"
        
        # Store in both old and new format for compatibility
        self.initialDomain = domain
        self.initialDomains[defaultDomainName] = domain

        # Create directories if needed
        initialDomainDir = os.path.join(self.projectPath, "domains", "initialDomain")
        os.makedirs(initialDomainDir, exist_ok=True)

        # Save the initial domain (keep original filename for backward compatibility)
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

        # Store in both old and new format for compatibility
        self.initialDomainPath = domainPath
        self.initialDomainPaths[defaultDomainName] = domainPath
        
        # Update project info with both old and new format
        self.updateProjectInfo("initialDomainPath", domainPath)
        self.updateProjectInfo("initialDomainPaths", self.initialDomainPaths)
        print(f"Initial domain and visualization meshes saved to {initialDomainDir}")
        return self

    def setTargetLevelSet(self, levelSet):
        """Set the target level set for the project (backward compatibility method)."""
        # Maintain backward compatibility by using a default target name
        defaultTargetName = "default"
        
        # Store in both old and new format for compatibility
        self.targetLevelSet = levelSet
        self.targetLevelSets[defaultTargetName] = levelSet

        # Create directories if needed
        targetDomainDir = os.path.join(self.projectPath, "domains", "targetDomain")
        os.makedirs(targetDomainDir, exist_ok=True)

        # Save the target level set (keep original filename for backward compatibility)
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

        # Store in both old and new format for compatibility
        self.targetLevelSetPath = domainPath
        self.targetLevelSetPaths[defaultTargetName] = domainPath
        
        # Update project info with both old and new format
        self.updateProjectInfo("targetLevelSetPath", domainPath)
        self.updateProjectInfo("targetLevelSetPaths", self.targetLevelSetPaths)
        print(f"Target level set and visualization meshes saved to {targetDomainDir}")
        return self

    def isReady(self):
        """Check if the project is ready for optimization/sensitivity studies."""
        # Check for initial domains - support both single and multi-domain scenarios
        hasInitialDomains = False
        if self.initialDomain is not None:
            hasInitialDomains = True
        elif len(self.initialDomains) > 0:
            hasInitialDomains = True
            
        if not hasInitialDomains:
            print("No initial domains are set.")
            return False
            
        # Check for targets - support both single and multi-target scenarios
        hasTargets = False
        if self.targetLevelSet is not None:
            hasTargets = True
        elif len(self.targetLevelSets) > 0:
            hasTargets = True
            
        if not hasTargets:
            print("No target level sets are set.")
            return False
            
        return True

    def isReadyForMultiTarget(self):
        """Check if the project is ready for multi-target optimization/sensitivity studies."""
        # Check for initial domains
        hasInitialDomains = False
        if self.initialDomain is not None:
            hasInitialDomains = True
        elif len(self.initialDomains) > 0:
            hasInitialDomains = True
            
        if not hasInitialDomains:
            print("No initial domains are set.")
            return False
            
        if len(self.targetLevelSets) == 0:
            print("No target level sets are set.")
            return False
        return True

    def isReadyForMultiDomain(self):
        """Check if the project is ready for multi-domain optimization/sensitivity studies."""
        if len(self.initialDomains) == 0:
            print("No initial domains are set.")
            return False
            
        # Check for targets
        hasTargets = False
        if self.targetLevelSet is not None:
            hasTargets = True
        elif len(self.targetLevelSets) > 0:
            hasTargets = True
            
        if not hasTargets:
            print("No target level sets are set.")
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

    def setInitialDomainFromFile(self, filePath: str):
        """
        Set initial domain from file and copy it to project structure.

        Args:
            filePath: Path to the domain file (.vpsd format)
        """
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"Domain file '{filePath}' does not exist.")

        if not filePath.endswith(".vpsd"):
            raise ValueError("Initial domain file must be in .vpsd format.")

        # Load the domain
        domain = Reader(filePath).apply()

        # Use existing setInitialDomain method to save to project structure
        self.setInitialDomain(domain)
        print(f"Initial domain set from '{filePath}' and saved to project")
        return self

    def setTargetDomainFromFile(self, filePath: str):
        """
        Set target domain from file and copy it to project structure.

        Args:
            filePath: Path to the level set file (.lvst format)
        """
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"Target domain file '{filePath}' does not exist.")

        if not filePath.endswith(".lvst"):
            raise ValueError("Target domain file must be in .lvst format.")

        # Load the level set
        levelSet = lsDomain()
        lsReader(levelSet, filePath).apply()

        # Use existing setTargetLevelSet method to save to project structure
        self.setTargetLevelSet(levelSet)
        print(f"Target domain set from '{filePath}' and saved to project")
        return self

    def addTargetLevelSet(self, name: str, levelSet):
        """Add a named target level set to the project."""
        if name in self.targetLevelSets:
            print(f"Warning: Target '{name}' already exists. Overwriting.")
        
        self.targetLevelSets[name] = levelSet

        # Create directories if needed
        targetDomainDir = os.path.join(self.projectPath, "domains", "targetDomain")
        os.makedirs(targetDomainDir, exist_ok=True)

        # Save the target level set with name-specific filename
        domainPath = os.path.join(
            targetDomainDir, f"{self.projectName}-targetDomain-{name}.lvst"
        )

        if self.mode == "2D":
            import viennals2d as vls

            # Save the level set
            vls.Writer(levelSet, domainPath).apply()

            # Create mesh visualization
            meshLS = vls.Mesh()
            vls.ToMesh(levelSet, meshLS).apply()
            meshPath = os.path.join(
                targetDomainDir, f"{self.projectName}-targetDomain-{name}-ls.vtp"
            )
            vls.VTKWriter(meshLS, meshPath).apply()

            # Create surface mesh visualization
            meshSurface = vls.Mesh()
            vls.ToSurfaceMesh(levelSet, meshSurface).apply()
            surfacePath = os.path.join(
                targetDomainDir, f"{self.projectName}-targetDomain-{name}-surface.vtp"
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
                targetDomainDir, f"{self.projectName}-targetDomain-{name}-ls.vtp"
            )
            vls.VTKWriter(meshLS, meshPath).apply()

            # Create surface mesh visualization
            meshSurface = vls.Mesh()
            vls.ToSurfaceMesh(levelSet, meshSurface).apply()
            surfacePath = os.path.join(
                targetDomainDir, f"{self.projectName}-targetDomain-{name}-surface.vtp"
            )
            vls.VTKWriter(meshSurface, surfacePath).apply()
        else:
            raise ValueError("Invalid mode. Only '2D' and '3D' are supported.")

        self.targetLevelSetPaths[name] = domainPath
        
        # Update project info with the multi-target paths
        self.updateProjectInfo("targetLevelSetPaths", self.targetLevelSetPaths)
        print(f"Target level set '{name}' and visualization meshes saved to {targetDomainDir}")
        return self

    def getTargetLevelSet(self, name: str):
        """Get a named target level set from the project."""
        if name not in self.targetLevelSets:
            raise KeyError(f"Target level set '{name}' not found.")
        return self.targetLevelSets[name]

    def getTargetLevelSetPath(self, name: str):
        """Get the file path for a named target level set."""
        if name not in self.targetLevelSetPaths:
            raise KeyError(f"Target level set path for '{name}' not found.")
        return self.targetLevelSetPaths[name]

    def removeTargetLevelSet(self, name: str):
        """Remove a named target level set from the project."""
        if name not in self.targetLevelSets:
            raise KeyError(f"Target level set '{name}' not found.")
        
        # Remove from memory
        del self.targetLevelSets[name]
        
        # Remove file path reference
        if name in self.targetLevelSetPaths:
            # Optionally remove the actual files
            filePath = self.targetLevelSetPaths[name]
            if os.path.exists(filePath):
                os.remove(filePath)
            del self.targetLevelSetPaths[name]
        
        # Update project info
        self.updateProjectInfo("targetLevelSetPaths", self.targetLevelSetPaths)
        print(f"Target level set '{name}' removed from project")
        return self

    def listTargetLevelSets(self):
        """List all target level set names in the project."""
        return list(self.targetLevelSets.keys())

    def getTargetLevelSetCount(self):
        """Get the number of target level sets in the project."""
        return len(self.targetLevelSets)

    def addTargetLevelSetFromFile(self, name: str, filePath: str):
        """Add a named target level set from file."""
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"Target domain file '{filePath}' does not exist.")

        if not filePath.endswith(".lvst"):
            raise ValueError("Target domain file must be in .lvst format.")

        # Load the level set
        levelSet = lsDomain()
        lsReader(levelSet, filePath).apply()

        # Use existing addTargetLevelSet method to save to project structure
        self.addTargetLevelSet(name, levelSet)
        print(f"Target domain '{name}' set from '{filePath}' and saved to project")
        return self

    def addInitialDomain(self, name: str, domain):
        """Add a named initial domain to the project."""
        if name in self.initialDomains:
            print(f"Warning: Initial domain '{name}' already exists. Overwriting.")
        
        self.initialDomains[name] = domain

        # Create directories if needed
        initialDomainDir = os.path.join(self.projectPath, "domains", "initialDomain")
        os.makedirs(initialDomainDir, exist_ok=True)

        # Save the initial domain with name-specific filename
        domainPath = os.path.join(
            initialDomainDir, f"{self.projectName}-initialDomain-{name}.vpsd"
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
                    initialDomainDir, f"{self.projectName}-initialDomain-{name}-ls.vtp"
                )
                vls.VTKWriter(meshLS, meshPath).apply()

                # Create surface mesh visualization
                meshSurface = vls.Mesh()
                vls.ToSurfaceMesh(lastLevelSet, meshSurface).apply()
                surfacePath = os.path.join(
                    initialDomainDir, f"{self.projectName}-initialDomain-{name}-surface.vtp"
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
                    initialDomainDir, f"{self.projectName}-initialDomain-{name}-ls.vtp"
                )
                vls.VTKWriter(meshLS, meshPath).apply()

                # Create surface mesh visualization
                meshSurface = vls.Mesh()
                vls.ToSurfaceMesh(lastLevelSet, meshSurface).apply()
                surfacePath = os.path.join(
                    initialDomainDir, f"{self.projectName}-initialDomain-{name}-surface.vtp"
                )
                vls.VTKWriter(meshSurface, surfacePath).apply()
        else:
            raise ValueError("Invalid mode. Only '2D' and '3D' are supported.")

        self.initialDomainPaths[name] = domainPath
        
        # Update project info with the multi-domain paths
        self.updateProjectInfo("initialDomainPaths", self.initialDomainPaths)
        print(f"Initial domain '{name}' and visualization meshes saved to {initialDomainDir}")
        return self

    def getInitialDomain(self, name: str):
        """Get a named initial domain from the project."""
        if name not in self.initialDomains:
            raise KeyError(f"Initial domain '{name}' not found.")
        return self.initialDomains[name]

    def getInitialDomainPath(self, name: str):
        """Get the file path for a named initial domain."""
        if name not in self.initialDomainPaths:
            raise KeyError(f"Initial domain path for '{name}' not found.")
        return self.initialDomainPaths[name]

    def removeInitialDomain(self, name: str):
        """Remove a named initial domain from the project."""
        if name not in self.initialDomains:
            raise KeyError(f"Initial domain '{name}' not found.")
        
        # Remove from memory
        del self.initialDomains[name]
        
        # Remove file path reference
        if name in self.initialDomainPaths:
            # Optionally remove the actual files
            filePath = self.initialDomainPaths[name]
            if os.path.exists(filePath):
                os.remove(filePath)
            del self.initialDomainPaths[name]
        
        # Update project info
        self.updateProjectInfo("initialDomainPaths", self.initialDomainPaths)
        print(f"Initial domain '{name}' removed from project")
        return self

    def listInitialDomains(self):
        """List all initial domain names in the project."""
        return list(self.initialDomains.keys())

    def getInitialDomainCount(self):
        """Get the number of initial domains in the project."""
        return len(self.initialDomains)

    def addInitialDomainFromFile(self, name: str, filePath: str):
        """Add a named initial domain from file."""
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"Initial domain file '{filePath}' does not exist.")

        if not filePath.endswith(".vpsd"):
            raise ValueError("Initial domain file must be in .vpsd format.")

        # Load the domain
        domain = Reader(filePath).apply()

        # Use existing addInitialDomain method to save to project structure
        self.addInitialDomain(name, domain)
        print(f"Initial domain '{name}' set from '{filePath}' and saved to project")
        return self
