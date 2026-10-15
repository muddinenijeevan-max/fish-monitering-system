# Fish Monitoring System

**Aquatic Robotics | Robot Simulation | Dynamics and Control**

A robotics project focused on developing and simulating an aquatic monitoring system for fish habitat environments. The project combines robot control, simulation integration, and aquatic tank-related utilities to explore the application of robotics in environmental monitoring.

## Overview

The Fish Monitoring System is designed to explore robotic approaches for aquatic environment observation and fish habitat monitoring.

The project incorporates Python-based robot control and simulation support, with Blender integration for visualization and robotics workflows.

The repository provides the project's core scripts and can serve as a foundation for further development of autonomous aquatic robots, environmental sensing, and intelligent monitoring.

## Objectives

- Develop and explore robotic control methods for aquatic monitoring.
- Support robotics simulation and visualization workflows.
- Integrate robot-related Python code with Blender.
- Establish a foundation for future aquatic environmental sensing and analysis.

## Features

- **Robot control:** Python scripts for robot control and related logic.
- **Blender integration:** A live bridge module for connecting the project with Blender.
- **Tank utilities:** Supporting functions for tank-related simulation workflows.
- **Project reporting:** A script for generating project reports.
- **Extensible architecture:** A starting point for adding sensors, autonomy, and environmental monitoring features.

## Technologies Used

- Python
- Blender
- Robotics dynamics and control concepts
- Simulation and visualization tools

## Repository Structure

```text
fish-monitoring-system/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── blender_live_bridge.py
├── robot_code.py
├── robot_final_report.py
└── tank_standout_calls.py
```

### File Description

| File | Purpose |
|---|---|
| `robot_code.py` | Main robot-related code and control logic |
| `blender_live_bridge.py` | Blender integration and live communication support |
| `robot_final_report.py` | Project report generation |
| `tank_standout_calls.py` | Tank-related simulation utilities |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Specifies files and folders excluded from Git tracking |

## Installation and Setup

### Prerequisites

- Python installed on your system.
- Blender, if required for the visualization or integration workflow.
- Git for cloning the repository.

### 1. Clone the repository

```bash
git clone https://github.com/muddinenijeevan-max/fish-monitering-system.git
cd fish-monitering-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure Blender-specific dependencies are installed or accessed through the appropriate Blender Python environment.

## Running the Project

To run the robot code, use:

```bash
python robot_code.py
```

For Blender-related functionality, open Blender and follow the integration workflow implemented in `blender_live_bridge.py`.

The exact execution sequence may depend on the simulation configuration, available assets, and required dependencies.

## Project Status

The repository contains the initial project source files and simulation-related modules.

Further development may include:

- Autonomous aquatic robot navigation.
- Integration of water-quality sensors.
- Environmental data collection and visualization.
- Intelligent fish habitat monitoring.
- Improved simulation and control capabilities.

## Contributors

This project was developed collaboratively by:

- **M. Jeevan**
- **G. Akhill**
- **P. Navadeep**

## License

No license has been specified yet. Please contact the project contributors for permission to reuse or distribute this project's code.
