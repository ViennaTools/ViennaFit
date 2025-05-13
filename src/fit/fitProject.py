import os
import json
import datetime


class Project:
    def __init__(self, projectName: str = "UnnamedProject"):
        self.projectName = projectName
        self.projectPath = f"./{projectName}"
        self.projectAbsPath = os.path.abspath(self.projectPath)

    def initialize(self):
        # Check if the project directory already exists
        if os.path.exists(self.projectPath):
            print(f"Project '{self.projectName}' already exists. Continuing with existing project.")
            # todo: load project state from json or similar
            return
        
        # Create the main project directory
        os.makedirs(self.projectPath, exist_ok=True)

        # Create subdirectories
        subdirs = ["annotations", "targets", "optRuns", "paramStudies", "hits"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.projectPath, subdir), exist_ok=True)

        print(f"Project '{self.projectName}' initialized with standard structure.")

        # save project info to json
        # Create project information dictionary
        projectInfo = {
            "projectName": self.projectName,
            "projectPath": self.projectAbsPath,
            "createdDate": str(datetime.datetime.now()),
            "subdirectories": ["annotations", "targets", "optRuns", "paramStudies", "hits"]
        }

        # Save project information to JSON file
        projectInfoPath = os.path.join(self.projectPath, "projectInfo.json")
        with open(projectInfoPath, "w") as f:
            json.dump(projectInfo, f, indent=4)

        print(f"Project information saved to {projectInfoPath}")


    def load(self, projectPath: str = "projectPath"):
        self.projectPath = projectPath
        if not os.path.exists(self.projectPath):
            print(f"Project '{self.projectName}' does not exist.")
            return

        # Load project info from json
        projectInfoPath = os.path.join(self.projectPath, "projectInfo.json")
        with open(projectInfoPath, "r") as f:
            projectInfo = json.load(f)

        print(f"Project '{self.projectName}' loaded with the following information:")
        print(json.dumps(projectInfo, indent=4))

