# Wheelchair-Friendly Navigation System Product Requirements Document

## Product Overview

This document outlines the requirements for a wheelchair-friendly navigation system that addresses the limitations of conventional navigation applications by specifically considering accessibility features crucial for wheelchair users. The system will implement pathfinding algorithms to determine optimal routes considering wheelchair accessibility factors.

## Problem Statement

Standard navigation applications like Google Maps provide basic route planning but fail to consider critical accessibility features including:

- Kerb ramps
- Sidewalk widths
- Slopes
- Indoor lifts
- Other wheelchair-specific obstacles and constraints

This creates significant challenges for wheelchair users trying to navigate through urban environments.

## Task Breakdown

### Task 1: Environment Creation and Problem Formation

1. **Environment Creation**:

   - Create a digital representation of a wheelchair-accessible environment with at least 20 path segments
   - Employ mapping tools (Google My Maps, OpenStreetMap, etc.) to annotate accessible path segments
   - Add markers for key amenities (wheelchair-accessible restrooms, parking spaces, shops, lifts, etc.)
   - Gather environmental information through available methods (surveys, sketches, online research, etc.)
   - Use reasonable figurative estimations if exact measurements are unavailable

2. **Problem Formation**:
   - Define the specific problem and environment parameters
   - Document justification for included features and tools used for map creation

### Task 2: Basic Navigation Implementation

1. **A\* Algorithm Implementation**:

   - Implement the A\* algorithm for wheelchair map navigation
   - Represent the environment as an adjacency matrix showing connections between locations and costs
   - Design appropriate heuristics for the environment

2. **Testing and Validation**:
   - Test implementation with different start and end points
   - Ensure returned paths are valid and optimal
   - Document agent and heuristic choices with justification
   - Provide annotated code explanations and implementation results

### Task 3: Enhanced Environment and Heuristics Comparison

1. **Environment Expansion**:

   - Add at least 10 more path segments (minimum total of 30 path segments)
   - Incorporate additional environmental constraints (kerb ramps, slopes, obstacles, etc.)

2. **Heuristic Enhancement**:
   - Create a new heuristic for pathfinding that accounts for the environmental constraints
   - Compare the new heuristic with the initial one from Task 2
   - Document the new heuristic's logic and impact on navigation results
   - Analyze performance and accuracy differences between both heuristics with supporting data

### Task 4: Performance Enhancement with Alternative Algorithm

1. **Alternative Algorithm Implementation**:

   - Implement an alternative pathfinding algorithm (e.g., Dijkstra's, DFS, Contraction Hierarchies, Transit Node Routing)
   - Integrate the alternative algorithm with the existing environment

2. **Performance Comparison**:
   - Compare the alternative algorithm's performance with A\* in terms of:
     - Efficiency (time complexity, execution speed)
     - Accuracy (optimality of paths)
     - Scalability (performance with increasing map size)
   - Evaluate suitability for wheelchair navigation under various conditions

### Task 5: Graphical User Interface

Create a graphical user interface with the following elements:

- User input for start and end points
- Map visualization of calculated paths and costs
- Display of additional path and environment information
- Summary of costs for each calculated route
- User-friendly design with a modern and aesthetically pleasing interface

## Technical Requirements

### Environment Representation

- Digital map with appropriate accessibility annotations
- Adjacency matrix for location connections and costs
- Representation of constraints and accessibility features

### Algorithm Implementation

- A\* pathfinding algorithm as the primary solution
- Custom heuristics for wheelchair accessibility
- At least one alternative algorithm for comparison
- Performance metrics for evaluation

### Code Requirements

- Every function must have:
  - A single responsibility
  - Type annotations
  - A docstring that includes:
    - A brief description of the function
    - Input arguments with types and descriptions
    - Return type(s) and description(s)
- Clearly define complex types by using dataclasses, TypedDict, NamedTuple, etc.
- Avoid deep nesting by using guard clauses for early returns and decomposing complex functions into smaller ones
- Avoid magic numbers by defining constants
- Write variable names in full (e.g. "exception Exception as error")
- Only write essential comments that explain the "Why", not the "What"
- Follow other professional coding standards and best practices
