import os
import json
import datetime
import csv
import shutil
from typing import List, Dict, Tuple, Optional, Any
from viennaps import Reader
from viennaps import Domain as psDomain
from viennals import Reader as lsReader
from viennals import Domain as lsDomain


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
            self.initialDomain = psDomain()
            # Load the initial domain
            Reader(
                self.initialDomain,
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
            dimension = 2  # ViennaLS target domains are always 2D
            targetDomain = lsDomain(dimension)
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
                dimension = 2  # ViennaLS target domains are always 2D
                targetDomain = lsDomain(dimension)
                lsReader(targetDomain, targetPath).apply()
                self.targetLevelSets[targetName] = targetDomain
                
                # If this is the default target and we don't have a single target loaded, use it
                if targetName == "default" and self.targetLevelSet is None:
                    self.targetLevelSet = targetDomain

        # Load all named initial domains (new multi-domain support)
        self.initialDomains = {}
        for domainName, domainPath in self.initialDomainPaths.items():
            if domainPath != "" and os.path.exists(domainPath):
                domain = psDomain()
                Reader(domain, domainPath).apply()
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
            import viennaps as vps
            import viennals as vls
            vps.setDimension(2)
            vls.setDimension(2)

            # Save the domain
            vps.Writer(domain, domainPath).apply()

            # Save the surface mesh
            domain.saveSurfaceMesh(os.path.join(
                initialDomainDir, f"{self.projectName}-initialDomain-surfaceMesh.vtp"
            ), True)

            domain.saveVolumeMesh(os.path.join(
                initialDomainDir, f"{self.projectName}-initialDomain-volumeMesh"
            ), True)

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
            import viennaps as vps
            import viennals as vls
            vps.setDimension(3)
            vls.setDimension(3)

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

        # ViennaLS target domains are always 2D
        import viennals as vls
        vls.setDimension(2)

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
        domain = psDomain()
        Reader(domain, filePath).apply()

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
        dimension = 2  # ViennaLS target domains are always 2D
        levelSet = lsDomain(dimension)
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

        # ViennaLS target domains are always 2D
        import viennals as vls
        vls.setDimension(2)

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
        dimension = 2  # ViennaLS target domains are always 2D
        levelSet = lsDomain(dimension)
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
            import viennaps as vps
            import viennals as vls
            vps.setDimension(2)
            vls.setDimension(2)

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
            import viennaps as vps
            import viennals as vls
            vps.setDimension(3)
            vls.setDimension(3)

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
        domain = psDomain()
        Reader(domain, filePath).apply()

        # Use existing addInitialDomain method to save to project structure
        self.addInitialDomain(name, domain)
        print(f"Initial domain '{name}' set from '{filePath}' and saved to project")
        return self

    def isMultiDomainReady(self) -> bool:
        """Check if project is ready for multi-domain optimization."""
        # Must have at least one initial domain and one target domain
        hasInitialDomains = len(self.initialDomains) > 0 or self.initialDomain is not None
        hasTargetDomains = len(self.targetLevelSets) > 0 or self.targetLevelSet is not None
        
        if not hasInitialDomains:
            return False
        if not hasTargetDomains:
            return False
            
        # For multi-domain, check that all initial domains have corresponding targets
        if len(self.initialDomains) > 0 and len(self.targetLevelSets) > 0:
            initialNames = set(self.initialDomains.keys())
            targetNames = set(self.targetLevelSets.keys())
            
            # All initial domains must have corresponding targets
            if not initialNames.issubset(targetNames):
                missingTargets = initialNames - targetNames
                print(f"Missing target domains for: {', '.join(missingTargets)}")
                return False
                
        return True
    
    def validateMultiDomainSetup(self) -> List[str]:
        """Validate multi-domain setup and return list of issues."""
        issues = []
        
        # Check initial domains
        if len(self.initialDomains) == 0 and self.initialDomain is None:
            issues.append("No initial domains are set")
        
        # Check target domains
        if len(self.targetLevelSets) == 0 and self.targetLevelSet is None:
            issues.append("No target domains are set")
        
        # Check name matching for multi-domain case
        if len(self.initialDomains) > 0 and len(self.targetLevelSets) > 0:
            initialNames = set(self.initialDomains.keys())
            targetNames = set(self.targetLevelSets.keys())
            
            # Check for missing targets
            missingTargets = initialNames - targetNames
            if missingTargets:
                issues.append(f"Missing target domains for initial domains: {', '.join(missingTargets)}")
            
            # Check for extra targets (not necessarily an error, but worth noting)
            extraTargets = targetNames - initialNames
            if extraTargets:
                issues.append(f"Extra target domains without corresponding initial domains: {', '.join(extraTargets)}")
        
        return issues
    
    def printMultiDomainStatus(self):
        """Print status of multi-domain setup."""
        print("Multi-Domain Setup Status:")
        print(f"  Initial domains: {len(self.initialDomains)} ({', '.join(self.initialDomains.keys()) if self.initialDomains else 'none'})")
        print(f"  Target domains: {len(self.targetLevelSets)} ({', '.join(self.targetLevelSets.keys()) if self.targetLevelSets else 'none'})")
        print(f"  Backward compatibility - Single initial domain: {'Yes' if self.initialDomain is not None else 'No'}")
        print(f"  Backward compatibility - Single target domain: {'Yes' if self.targetLevelSet is not None else 'No'}")
        
        issues = self.validateMultiDomainSetup()
        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✓ Multi-domain setup is valid")
            
        print(f"  Multi-domain ready: {'Yes' if self.isMultiDomainReady() else 'No'}")
    
    def getDomainPairings(self) -> Dict[str, Tuple[str, str]]:
        """Get dictionary of domain name -> (initial_path, target_path) pairings."""
        pairings = {}

        if len(self.initialDomains) > 0 and len(self.targetLevelSets) > 0:
            # Multi-domain case
            for domainName in self.initialDomains.keys():
                if domainName in self.targetLevelSets:
                    initialPath = self.initialDomainPaths.get(domainName, "")
                    targetPath = self.targetLevelSetPaths.get(domainName, "")
                    pairings[domainName] = (initialPath, targetPath)
        else:
            # Single domain case (backward compatibility)
            if self.initialDomain is not None and self.targetLevelSet is not None:
                pairings["default"] = (self.initialDomainPath, self.targetLevelSetPath)

        return pairings

    def getOptimizationSummary(self, forceRescan: bool = False):
        """
        Get optimization runs summary for this project.

        Args:
            forceRescan: If True, re-scan all runs instead of loading from cached summary

        Returns:
            RunsSummary object with aggregated optimization results
        """
        from .fitRunsSummary import RunsSummary

        summary = RunsSummary(self.projectAbsPath)

        if forceRescan:
            summary.scanRuns()
        else:
            # Try to load existing summary, fall back to scanning if not found
            if not summary.loadSummary():
                summary.scanRuns()

        return summary

    def printOptimizationOverview(self):
        """Print a formatted overview of all optimization runs in this project."""
        summary = self.getOptimizationSummary()
        summary.printSummary()

    def getBestOptimizationRun(self) -> Optional[Dict]:
        """
        Get information about the best optimization run in this project.

        Returns:
            Dictionary with best run information, or None if no runs found
        """
        summary = self.getOptimizationSummary()
        return summary.getBestRun()

    def updateOptimizationSummary(self):
        """
        Update the optimization summary file by re-scanning all runs.
        This should be called after new optimization runs complete.
        """
        summary = self.getOptimizationSummary(forceRescan=True)
        summary.saveSummary()
        print(f"Optimization summary updated for project {self.projectName}")

    def _loadIncompleteRunData(
        self, runDir: str, runName: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load data from incomplete optimization run files.

        Args:
            runDir: Path to the run directory
            runName: Name of the run

        Returns:
            Dictionary containing:
            - bestParameters: dict[str, float]
            - bestScore: float
            - bestEvaluationNumber: int
            - actualNumEvaluations: int
            - fixedParameters: dict[str, float]
            - variableParameters: dict[str, tuple[float, float]]
            - optimizer: str
            Returns None if critical files are missing.
        """
        # Define reserved column names that are not parameters
        reservedColumns = {
            "evaluationNumber",
            "elapsedTime",
            "simulationTime",
            "distanceMetricTime",
            "totalElapsedTime",
            "objectiveValue",
        }

        # Load progressBest.csv to get best evaluation data
        progressBestPath = os.path.join(runDir, "progressBest.csv")
        if not os.path.exists(progressBestPath):
            print(f"Error: progressBest.csv not found in {runDir}")
            return None

        try:
            with open(progressBestPath, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    print(f"Error: progressBest.csv is empty in {runDir}")
                    return None

                # Get last row (best evaluation so far)
                lastRow = rows[-1]

                # Extract basic data
                bestScore = float(lastRow["objectiveValue"])
                bestEvaluationNumber = int(lastRow["evaluationNumber"])

                # Extract parameters (all columns except reserved ones)
                bestParameters = {}
                for key, value in lastRow.items():
                    if key not in reservedColumns and not key.endswith(
                        "_value"
                    ) and not key.endswith("_time"):
                        try:
                            bestParameters[key] = float(value)
                        except (ValueError, TypeError):
                            pass  # Skip non-numeric values

        except Exception as e:
            print(f"Error reading progressBest.csv: {e}")
            return None

        # Load progressAll.csv to count actual evaluations
        progressAllPath = os.path.join(runDir, "progressAll.csv")
        if not os.path.exists(progressAllPath):
            print(f"Error: progressAll.csv not found in {runDir}")
            return None

        try:
            with open(progressAllPath, "r") as f:
                reader = csv.DictReader(f)
                actualNumEvaluations = len(list(reader))
        except Exception as e:
            print(f"Error reading progressAll.csv: {e}")
            return None

        # Load metadata (try progressAll_metadata.json first, then startingConfiguration.json)
        fixedParameters = {}
        variableParameters = {}
        optimizer = "unknown"

        metadataPath = os.path.join(runDir, "progressAll_metadata.json")
        if os.path.exists(metadataPath):
            try:
                with open(metadataPath, "r") as f:
                    metadata = json.load(f)
                    fixedParameters = metadata.get("fixedParameters", {})
                    # variableParameters stored as parameterBounds in metadata
                    paramBounds = metadata.get("parameterBounds", {})
                    for paramName, bounds in paramBounds.items():
                        if isinstance(bounds, list) and len(bounds) == 2:
                            variableParameters[paramName] = tuple(bounds)
                    optimizer = metadata.get("optimizer", "unknown")
            except Exception as e:
                print(f"Warning: Could not load progressAll_metadata.json: {e}")

        # Fallback to startingConfiguration.json if metadata not loaded
        if not variableParameters:
            configPath = os.path.join(runDir, f"{runName}-startingConfiguration.json")
            if os.path.exists(configPath):
                try:
                    with open(configPath, "r") as f:
                        config = json.load(f)
                        fixedParameters = config.get("fixedParameters", {})
                        variableParameters = config.get("variableParameters", {})
                        # Convert lists to tuples if needed
                        for key, value in variableParameters.items():
                            if isinstance(value, list) and len(value) == 2:
                                variableParameters[key] = tuple(value)
                        optimizer = config.get("optimizer", "unknown")
                except Exception as e:
                    print(f"Warning: Could not load startingConfiguration.json: {e}")

        return {
            "bestParameters": bestParameters,
            "bestScore": bestScore,
            "bestEvaluationNumber": bestEvaluationNumber,
            "actualNumEvaluations": actualNumEvaluations,
            "fixedParameters": fixedParameters,
            "variableParameters": variableParameters,
            "optimizer": optimizer,
        }

    def finalizeOptimizationRun(
        self, runName: str, copyDomainFiles: bool = True
    ) -> bool:
        """
        Finalize an incomplete optimization run.

        Creates final-results.json using the best evaluation found so far,
        copies domain files to optimalDomains, and updates optimization summary.

        Args:
            runName: Name of the optimization run to finalize
            copyDomainFiles: Whether to copy best domain files to optimalDomains folder (default: True)

        Returns:
            True if finalization succeeded, False otherwise

        Example:
            >>> project = Project()
            >>> project.load("./myProject")
            >>> project.finalizeOptimizationRun("incomplete_run_1")
        """
        # Validation Phase
        runDir = os.path.join(self.projectPath, "optimizationRuns", runName)

        if not os.path.exists(runDir):
            print(f"Error: Run directory not found: {runDir}")
            return False

        # Check if already finalized
        finalResultsPath = os.path.join(runDir, f"{runName}-final-results.json")
        if os.path.exists(finalResultsPath):
            print(
                f"Warning: Run '{runName}' is already finalized. Skipping finalization."
            )
            return True  # Return True since it's already in desired state

        # Verify progress files exist
        progressBestPath = os.path.join(runDir, "progressBest.csv")
        progressAllPath = os.path.join(runDir, "progressAll.csv")

        if not os.path.exists(progressBestPath):
            print(
                f"Error: Cannot finalize run '{runName}' - progressBest.csv not found."
            )
            print(f"Expected location: {progressBestPath}")
            return False

        if not os.path.exists(progressAllPath):
            print(
                f"Error: Cannot finalize run '{runName}' - progressAll.csv not found."
            )
            print(f"Expected location: {progressAllPath}")
            return False

        # Data Loading Phase
        print(f"Finalizing optimization run '{runName}'...")
        runData = self._loadIncompleteRunData(runDir, runName)

        if runData is None:
            print(f"Error: Failed to load data from incomplete run '{runName}'")
            return False

        # Check if any evaluations were completed
        if runData["actualNumEvaluations"] == 0:
            print(
                f"Error: Cannot finalize run '{runName}' - no evaluations completed."
            )
            return False

        # Results Creation Phase
        result = {
            "bestScore": runData["bestScore"],
            "bestEvaluation#": runData["bestEvaluationNumber"],
            "bestParameters": runData["bestParameters"],
            "fixedParameters": runData["fixedParameters"],
            "variableParameters": runData["variableParameters"],
            "optimizer": runData["optimizer"],
            "numEvaluations": runData["actualNumEvaluations"],
        }

        try:
            with open(finalResultsPath, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Created final-results.json with {runData['actualNumEvaluations']} evaluations")
        except Exception as e:
            print(f"Error: Failed to create final-results.json: {e}")
            return False

        # Domain File Copy Phase
        if copyDomainFiles:
            projectDomainDir = os.path.join(self.projectAbsPath, "domains", "optimalDomains")
            os.makedirs(projectDomainDir, exist_ok=True)

            # Pattern matches both single and multi-domain files:
            # - Single: {name}-{eval:03d}.vtp
            # - Multi:  {name}-{eval:03d}-{domainName}.vtp
            basePattern = f"{runName}-{runData['bestEvaluationNumber']:03d}"
            progressDir = os.path.join(runDir, "progress")

            copiedFiles = []
            if os.path.exists(progressDir):
                for filename in os.listdir(progressDir):
                    if filename.startswith(basePattern) and filename.endswith(".vtp"):
                        sourcePath = os.path.join(progressDir, filename)
                        targetPath = os.path.join(projectDomainDir, filename)
                        try:
                            shutil.copy2(sourcePath, targetPath)
                            copiedFiles.append(filename)
                        except Exception as e:
                            print(f"Warning: Failed to copy {filename}: {e}")

            if copiedFiles:
                print(f"Copied {len(copiedFiles)} domain file(s) to {projectDomainDir}:")
                for filename in copiedFiles:
                    print(f"  - {filename}")
            else:
                print(
                    f"Warning: No domain files found matching pattern '{basePattern}*.vtp' in progress directory"
                )

        # Summary Update Phase
        try:
            self.updateOptimizationSummary()
        except Exception as e:
            print(f"Warning: Could not update optimization summary: {e}")

        # Success
        print(
            f"Successfully finalized run '{runName}' with best score {runData['bestScore']:.6f} "
            f"from evaluation #{runData['bestEvaluationNumber']}"
        )
        return True

    def generateOptimizationReport(self, outputPath: Optional[str] = None) -> str:
        """
        Generate a markdown report of all optimization runs.

        Args:
            outputPath: Optional path to save the report. If None, saves to project directory.

        Returns:
            Path to the generated report file
        """
        summary = self.getOptimizationSummary(forceRescan=True)

        if outputPath is None:
            outputPath = os.path.join(
                self.projectAbsPath, f"{self.projectName}-optimization-report.md"
            )

        return summary.generateMarkdownReport(outputPath)
