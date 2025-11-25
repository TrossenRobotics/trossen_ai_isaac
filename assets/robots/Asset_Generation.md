# Asset Generation Documentation

This document describes the provenance, generation process, and parameters used for creating the USD robot models in this repository.

## Table of Contents
- [Overview](#overview)
- [Source Models](#source-models)
- [USD Generation Process](#usd-generation-process)
- [Version History](#version-history)
- [Additional Resources](#additional-resources)

---

## Overview

All USD files in this directory were generated for NVIDIA Isaac Sim 5.1+ simulation. These assets represent the Trossen Robotics AI manipulator family in various configurations.

**Generated Assets:**
- `widowx_ai_base.usd` - Base configuration single arm
- `widowx_ai_follower.usd` - Follower configuration single arm
- `widowx_ai_leader_left.usd` - Left leader arm for teleoperation
- `widowx_ai_leader_right.usd` - Right leader arm for teleoperation
- `mobile_ai.usd` - Dual-arm mobile manipulator platform
- `stationary_ai.usd` - Dual-arm stationary platform

---

## Source Models

### URDF and Mesh Files

All USD files are generated from URDF descriptions and mesh files maintained in the official Trossen Robotics repository:

**Source Repository:** [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)

Each USD robot model (`widowx_ai_base.usd`, `widowx_ai_follower.usd`, `widowx_ai_leader_left.usd`, `widowx_ai_leader_right.usd`, `mobile_ai.usd`, `stationary_ai.usd`) corresponds to a URDF file in the source repository with the same naming convention.

**Mesh Files:**
- Visual and collision meshes are STL format
- Located in the `trossen_arm_description` repository under `meshes/`

**Units:** Standard SI units (meters, kilograms, radians)

**Coordinate System:** Z-up (Isaac Sim standard)

### Robot Parameters

All kinematic parameters and mass properties are based on real-world hardware specifications:
- **Kinematic Parameters:** Match actual robot parameters and link transforms
- **Mass Properties:** Based on real robot component masses and inertias
- **Joint Limits:** Match physical hardware joint limits
- **Mesh Geometry:** Generated from actual robot CAD models

---

## USD Generation Process

### Prerequisites

1. **Install Isaac Sim 5.1+**
   ```bash
   # Verify installation
   ~/isaacsim5.1/isaac-sim.sh --version
   ```

2. **Obtain Source Files**
   - Clone URDF files from [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)
   - Ensure mesh files are available from the same repository
   - Verify all mesh paths in URDF are relative

### Generation Steps

1. **Launch Isaac Sim:**
   ```bash
   cd ~/trossen_ai_isaac/robots
   ~/isaacsim5.1/isaac-sim.sh
   ```

2. **Import URDF:**
   - `File` → `Import`
   - Select URDF: `path/to/trossen_arm_description/urdf/wxai_base.urdf`

3. **Configure Import Settings:**
   ```
   ✓ Fix Base Link: False (mobile robots) / True (stationary arms)
   ✓ Ignore Mimic: False
   ✓ Joint Drive Type: Force
   ✓ Joint Drive Stiffness: [Use PID parameters from trossen_arm repo]*
   ✓ Joint Drive Damping: [Use PID parameters from trossen_arm repo]*
   ✓ Density: 0.0 (use mass from URDF)
   ✓ Distance Scale: 1.0 (SI units - meters)
   ✓ Self Collision: True
   ✓ Convex Decomposition: Enabled
   ```
   
   **PID Parameter Sources:**
   - Configuration File: [default_configurations_wxai_v0.yaml](https://github.com/TrossenRobotics/trossen_arm/blob/main/demos/python/default_configurations_wxai_v0.yaml#L191)
   - Documentation: [Motor Parameters](https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html#motor-parameters)

4. **Post-Import Refinement:**
   - **Visuals:** Check mesh rendering, tree structure, and all links/joints
   - **Collision:** Inspect and refine collision meshes, verify convex decomposition
   - **Joints:** Verify drive parameters, limits, and mimic joint configuration
   - **Textures:** Add textures and materials for realistic appearance
   - **Sensors:** Add camera at camera-specific frame with appropriate parameters
   - **Paths:** Ensure all mesh paths are relative

5. **Save USD:**
   - `File` → `Save As...` → `widowx_ai_base.usd`
   - Use relative paths for all references

6. **Validate:**
   - Test loading: `~/isaacsim5.1/isaac-sim.sh scripts/robot_bringup.py wxai_base`
   - Check console for errors (missing meshes, articulation issues)

   - Compare with previous version (if regenerating)

7. **Repeat** for other robot configurations

---

**Note:** All USD files use parameters matching real-world hardware specifications. Import date: November 25, 2025 (Isaac Sim 5.1.0).

---

## Version History

### Version Tracking

We use semantic versioning for USD assets: `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes (joint hierarchy, coordinate system changes)
- **MINOR:** New features (added sensors, updated meshes)
- **PATCH:** Bug fixes (corrected parameters, fixed paths)

### Changelog

#### v1.0.0 - November 25, 2025
**Initial Release**
- Generated from URDF files in [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)
- Isaac Sim version: 5.1.0
- Added 6 robot configurations
- Import method: Direct file import using Isaac Sim URDF Importer
- Drive type: Force (matching real-world hardware)
- Self-collision enabled with convex decomposition

**Assets:**
- `widowx_ai_base.usd` v1.0.0
- `widowx_ai_follower.usd` v1.0.0
- `widowx_ai_leader_left.usd` v1.0.0
- `widowx_ai_leader_right.usd` v1.0.0
- `mobile_ai.usd` v1.0.0
- `stationary_ai.usd` v1.0.0

**Features:**
- Real-world hardware parameter matching
- Camera sensors added at camera-specific frames
- Textures applied for visual realism
- Refined collision geometry
- Validated joint limits and mimic joints

#### Future Versions

**v1.1.0 (Planned)**
- Implement USD referencing/composition for robot variants (most robots are modified versions of base models)
- Enhance collision mesh accuracy and performance
- Refine textures to match real-world appearance more closely
- Add official Trossen Robotics logo texture to robot models

**v1.0.1 (Planned)**
- Add surface friction parameters to contact materials
- Minor bug fixes and parameter refinements

---

## Additional Resources

### Isaac Sim Documentation
- [URDF Importer](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html)

### Related Repositories
- **URDF & Meshes:** [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description) - Robot URDF descriptions and mesh files
- **PID Parameters:** [TrossenRobotics/trossen_arm](https://github.com/TrossenRobotics/trossen_arm) - Real-world motor PID control parameters used for joint drive stiffness and damping
  - Configuration: [default_configurations_wxai_v0.yaml](https://github.com/TrossenRobotics/trossen_arm/blob/main/demos/python/default_configurations_wxai_v0.yaml#L191)
  - Documentation: [Motor Parameters](https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html#motor-parameters)

### Contact
For questions about asset generation:
- **Repository Issues:** https://github.com/TrossenRobotics/trossen_ai_isaac/issues
- **Trossen Support:** support@trossenrobotics.com

---

**Document Version:** 1.0.0  
**Last Updated:** 11/25/2025 
**Maintained By:** Abhishek Chothani, abhishek.chothani@trossenrobotics.com
