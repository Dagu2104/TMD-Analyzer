# TMD-Analyzer Software.

## Description

Interactive program for the optimization of Tuned Mass Damper (TMD) dynamic parameters under a given earthquake ground-motion record.


## Installation Requirements

Not required. Access between streamlit link.

## Functionalities
### Side Bar
In the side bar the user can do the following action:
- Input all structural characteristics, such as spans, bays, number of stories, story heights, member sections, and floor weights. (It is possible to define specific heights or distances for  floors and spans respectively).
- In addition, the program allows the user to choose the beam formulation, either Euler–Bernoulli or Timoshenko beam theory.
<p align="center">
  <img src="./VISUAL README/SIDE BAR.png" alt="Side Bar" width="300" height="800" />
</p>

### General Data
In the first tab, labeled "General Data" is allowed:

- Plot the entire structure with the provided data.

<p align="center">
  <img src="./VISUAL README/PLOT_STRUCTURE.png" alt="Plot structure" width="900" height="1400" />
</p>

- The program displays the structure’s dynamic properties, including natural periods, angular frequencies, modal mass participation ratios (%), and cumulative mass participation ratios (%).
<p align="center">
  <img src="./VISUAL README/DYNAMIC_PROP.png" alt="TABLE DYNAMIC PROP" width="1200" height="1200" />
</p>

### Seismic Record Data
In the second tab, labeled "Seismic Record Data" the user can:

- Upload a seismic ground-motion record in PEER format.

<p align="center">
  <img src="./VISUAL README/PEER_FORMAT.png" alt="PEER FORMAT" width="800" height="1200" />
</p>

- The program automatically detects the time step and the number of data points. The program internally inserts an initial data point at t = 0 seconds with a corresponding acceleration of 0 g, ensuring proper initialization of the ground-motion record.
- Plot the PEER seismic record data.

<p align="center">
  <img src="./VISUAL README/SEISMIC_RECORD.png" alt="SEISMIC RECORD" width="800" height="1200" />
</p>

- Specificate the target frequency range for band-pass filtering of the seismic ground-motion record.
- Plot the filtered seismic ground-motion record.

<p align="center">
  <img src="./VISUAL README/FILTERED_SEISMIC_RECORD.png" alt="FILTERED SEISMIC RECORD" width="800" height="1200" />
</p>

### Results

In the third tab, the program displays the following information:

- Display a table of vibration modes for the structure without the TMD, including modal frequencies and mass participation ratios, up to a cumulative mass participation of 90%. 
- Also display a plot showing the reduction in interstory drift ratios achieved by the TMD

<p align="center">
  <img src="./VISUAL README/RESULTS_CLOUD.png" alt="RESULTS CLOUD" width="800" height="1200" />
</p>

- Display a table summarizing the interstory drift ratio reduction (IDRR), obtained from the comparison of peak interstory drift ratios between the controlled (with TMD) and uncontrolled (without TMD) structural responses, thereby reflecting the global effectiveness of the TMD.

<p align="center">
  <img src="./VISUAL README/OPTIMAL_TMD_PARAMETERS.png" alt="OPTIMAL TMD PARAMETERS" width="800" height="1200" />
</p>

### Animation Tab

In this tab, the user can observe the structural response to a seismic record and compare the behavior of the same structure with and without the TMD.

<p align="center">
  <img src="./VISUAL README/ANIMATION.png" alt="ANIMATION" width="800" height="800" />
</p>

- At bottom of the page, the program offers two plots:
  - The first figure illustrates the maximum absolute peak interstory drift ratios derived from the time-history analysis. The second figure displays the corresponding maximum absolute peak floor accelerations
 
<p align="center">
  <img src="./VISUAL README/DRIFT_ACCELERATION.png" alt="PLOTS DRIFT ACCELERATION" width="800" height="800" />
</p>




You can view and learn how to use the software video at the following link:
[Video](https://puceeduec-my.sharepoint.com/:v:/g/personal/meriverabo_puce_edu_ec/EQW5ooVRLFlFh4xyy14kCcYBj21V6zmJlNoDj3jcgiUeXg?e=AQzmLx&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)
