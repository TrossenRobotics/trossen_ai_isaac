# Asset Generation

## Overview

This document describes the generation process of the Universal Scene Description (USD) file for Trossen AI robots.

Generated Assets:

- `wxai_base.usd` - Base configuration single arm
- `wxai_follower.usd` - Follower configuration single arm
- `wxai_leader_left.usd` - Left leader arm for teleoperation
- `wxai_leader_right.usd` - Right leader arm for teleoperation
- `mobile_ai.usd` - Dual-arm mobile manipulator platform
- `stationary_ai.usd` - Dual-arm stationary platform

## Source Models

### URDF and Mesh Files

All USD files are generated from URDF descriptions maintained in the official Trossen Robotics repository: [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)

All kinematic parameters and mass properties are based on real-world hardware specifications

## USD Generation Process

### Prerequisites

1. Install Isaac Sim 5.1+

```bash
# Verify installation
~/isaacsim5.1/isaac-sim.sh --version
```

2. Obtain URDF Files

   - Clone URDF files from [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)

### Generation Steps

1. Launch Isaac Sim:

```bash
cd ~/trossen_ai_isaac/robots
~/isaacsim5.1/isaac-sim.sh
```

2. Import URDF:

   - `File` → `Import`
   - Select URDF: `path/to/trossen_arm_description/urdf/wxai_base.urdf`

3. Configure Import Settings:

   - Fix Base Link: False (mobile robots) / True (stationary arms)
   - Ignore Mimic: False
   - Joint Drive Type: Force
   - Joint Drive Stiffness: [Use PID parameters from trossen_arm repo]
   - Joint Drive Damping: [Use PID parameters from trossen_arm repo]
   - Density: 0.0 (use mass from URDF)
   - Distance Scale: 1.0 (SI units - meters)
   - Self Collision: True
   - Convex Decomposition: Enabled

   PID Parameter Sources:

   - Configuration File: [default_configurations_wxai_v0.yaml](https://github.com/TrossenRobotics/trossen_arm/blob/main/demos/python/default_configurations_wxai_v0.yaml#L191)
   - Documentation: [Motor Parameters](https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html#motor-parameters)

4. Post-Import Refinement:

   - Visuals: Check mesh rendering, tree structure, and all links/joints
   - Collision: Inspect and refine collision meshes if needed
   - Joints: Verify drive parameters, limits, and mimic joint configuration
   - Textures: Add textures and materials for realistic appearance
   - Sensors: Add camera at camera-specific frame with appropriate parameters
   - Paths: Ensure all mesh paths are relative

5. Save USD:

   - `File` → `Save As...` → `wxai_base.usd`
   - Use relative paths for all references

7. Validate:

   - Test loading: `~/isaacsim5.1/isaac-sim.sh scripts/robot_bringup.py wxai_base`
   - Check console for errors (missing meshes, articulation issues)

Note: All USD files use parameters matching real-world hardware specifications as of November 25, 2025, using Isaac Sim 5.1.0.

## Changelog

Initial Release

- Generated from URDF files in [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)
- Isaac Sim version: 5.1.0
- Added 6 robot configurations
- Import method: Direct file import using Isaac Sim URDF Importer

Future Versions

- Implement USD referencing/composition for robot variants (most robots are modified versions of base models)
- Refine textures to match real-world appearance more closely
- Add surface friction parameters to contact materials
