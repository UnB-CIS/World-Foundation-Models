### English

# Project Scope - World Foundation Model

## Project Overview

Develop, through 2D physical simulations, a World Foundation Model capable of learning the dynamics and kinematics of objects from visual observations (video/images) and predicting/planning the system’s future evolutions.  
The user describes the scene and the task through a prompt; the system instantiates the simulation, executes/predicts rollouts, and returns video outputs, metrics, and planned actions.

## Problem

- Team members currently lack practical experience with **world model architectures**, making this project necessary to enable them to gain expertise and understand both the theoretical foundations and the practical implementation of generative models applied to physical simulation of object behavior in different user-defined scenarios.
- There is still no unified, data-driven model that can:
  - **Generalize** to new combinations of objects/properties (mass, friction, elasticity) in 2D.
  - **Predict** future states with physical consistency across dozens of steps.
  - **Plan actions** to achieve specific goals (e.g., _“make the red block touch the target”_) without requiring task-specific hyperparameter tuning.

## Objectives

- Develop a functional **World Model** capable of interpreting textual descriptions (prompts) and transforming them into consistent and realistic 2D physical simulations.
- Produce an **academic paper** to consolidate the knowledge acquired throughout the project, contributing to scientific dissemination and analysis of the subject.
- Provide team members with a deeper understanding of emerging **artificial intelligence architectures**, expanding the group’s capacity for innovation and enabling the creation of educational, research, and technological development opportunities.

## Project Boundaries

- Train and evaluate a World Model capable of understanding and reproducing the kinematics and dynamics of objects in 2D simulations.
- The project **does not cover** 3D simulations or direct applications in physical environments, maintaining its focus on controlled digital environments.

## Out of Scope

- Application in **3D environments**.
- **End-to-end reinforcement learning** directly in hardware.
- **Deformable contact/complex fluids**; **competitive multi-agent** settings.

## Users / Stakeholders

- Members of the development and research team directly involved in the project.
- The academic and technical community interested in **World Models**, through the review article to be published.

## Functional Requirements

### The simulation environment must be able to:

- Generate 2D videos in formats such as `.MP4`, `.GIF`, or `.MKV`, representing object dynamics based on the trained model.
- Receive **natural language prompts** specifying elements, their properties (mass, color, shape, weight, etc.), initial conditions (position, velocity, potential energy, etc.), and interactions within the simulation.
- Allow multiple scenarios with different object configurations to be executed.
- Automatically log conducted experiments, saving results and metadata.

### The trained model must be able to:

- Infer **kinematics** (position, velocity, acceleration) and **dynamics** (forces, collisions, interactions) of simulated objects.
- **Generalize** to different scenarios, not limited to those present in training, provided they follow the same physical laws.
- Provide **performance metrics**, such as average trajectory prediction error.
- Interpret and simulate interactions between **2 to N simultaneous objects**, where `N <= 3`.

## Success Metrics

- Trajectory prediction metric (probability of following the calculated path).

## Documentation Topics Still to be Addressed

- Description of the architecture, techniques used, and rationale for design choices.
- Tutorials for reproducibility (environment setup, model usage, and example prompts).

## Non-Functional Requirements

Non-functional requirements refer to aspects not directly tied to the software’s functional behavior, but rather to quality attributes and constraints.

### Infrastructure

- The model must be trainable and executable on the computational infrastructure available in the lab (local CPU/GPU).
- A **lightweight version** should be available for execution on resource-constrained machines for testing purposes.

### Quality

- The simulation environment must present visual and physical consistency, without critical failures that would prevent analysis.
- The code must follow **software engineering best practices** (modularity, version control, basic unit testing).

### Usability

- The user interface must be simple and interactive, not requiring advanced programming knowledge for basic scenarios.
- Prompts should be written in clear natural language, without complex syntax requirements.

### Reproducibility

- The GitHub repository must contain complete instructions for installation, configuration, and execution.
- Experiments should be reproducible by third parties with access to the dataset and code.

## Deliverables

- **Frontend** that allows users to interact with the system and provide prompts.
- **Video outputs** representing the final result of the simulation generated from the user’s input.

## Risks & Assumptions

- Limited computational resources.
- Potential difficulties managing team workload.