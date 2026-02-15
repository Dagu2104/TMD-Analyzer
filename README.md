 # TMD-Analyzer Software.

## Description

Interactive program for the optimization of Tuned Mass Damper (TMD) dynamic parameters under a given earthquake ground-motion record.


## Installation Requirements

Not required. Access between streamlit link.

## Functionalities
In the side bar the user can do the following action:
- Input all structural characteristics, such as spans, bays, number of stories, story heights, member sections, and floor weights. (It is possible to define specific heights o distances for  floors and spans respectively).

<p align="center">
  <img src="./VISUAL README/SIDE BAR.png" alt="Side Bar" width="300" height="800" />
</p>

In the first tab, labeled "General Data" is allowed:

- Plot the entire structure with the provided data.
<p align="center">
  <img src="./VISUAL README/PLOT_STRUCTURE.png" alt="Plot structure" width="900" height="1400" />
</p>
- The program displays the structure’s dynamic properties, including natural periods, angular frequencies, modal mass participation ratios (%), and cumulative mass participation ratios.
<p align="center">
  <img src="./VISUAL README/DYNAMIC_PROP.png" alt="TABLE DYNAMIC PROP" width="1000" height="1200" />
</p>
In the second tab, labeled "Seismic Recor Data" the user can:
- Upload a seismic ground-motion record in PEER format. (
- The program automatically detects the time step and the number of data points.

## Installer

- Download the software installer by clicking on the following image:

<p align="center">
  <a href="https://puceeduec-my.sharepoint.com/:u:/g/personal/meriverabo_puce_edu_ec/EfxxPUOhf5dAi91r6gSuohgBJ_zwU8Uq53lI_8E1WxLG7g?download=1">
    <img src="./visual/download_dynamic_seism.png" alt="Descargar Dynamic Seism" width="600" height="300" />
  </a>
</p>



<!--
https://puceeduec-my.sharepoint.com/:u:/g/personal/meriverabo_puce_edu_ec/EX8Twoe_aGpIqtDKmPnoF6MBTavRt1THmwoqVWpqfrq4Bg?download=1
-->


## How Install and Configurate the External Tool wkhtmltopdf?
- Download the wkhtmltopdf tool by clicking on the following image:

<p align="center">
  <a href="https://puceeduec-my.sharepoint.com/:u:/g/personal/meriverabo_puce_edu_ec/EalVAyNhTORAsnbwS9TlDdIBuBpOhMn18fjeHBAWIlxfNA?download=1">
    <img src="./visual/wkhtml_image.png" alt="Administradorwkhtmltopdf" width="457.5" height="225" />
  </a>
</p>

<!--
https://puceeduec-my.sharepoint.com/:u:/g/personal/meriverabo_puce_edu_ec/EalVAyNhTORAsnbwS9TlDdIBuBpOhMn18fjeHBAWIlxfNA?download=1
-->

<div align="center">

| Name                  | Description                                                                                   | Website                                                      |
|-----------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| wkhtmltopdf           | Convert the files from HTML format to PDF                                                      | [https://wkhtmltopdf.org/index.html](https://wkhtmltopdf.org/index.html) |

</div>

- Run the wkhtmltopdf tool as an administrator and install it

<p align="center">
  <img src="./visual/Ejecutar_como_administrador_wkhtmltopdf.png" alt="Administradorwkhtmltopdf" />
</p>

- Read and accept the license agreement, then click 'I Agree'

<p align="center">
  <img src="./visual/Acuerdo_de_Licencia_de_la_herramienta_wkhtmltopdf.png" alt="Acuerdowkhtmltopdf" />
</p>

- Choose an installation path, then click "Install"

<p align="center">
<img src="./visual/ruta_de_instalacion_de_la_herramienta_wkhtmltopdf.png" alt="Eutawkhtmpltopdf" />
</p>

- Once the installation is complete, click 'Close'
<p align="center">
<img src="./visual/finalizacion_de_la_instalacion.png" alt="Finalizacionwkhtmltopdf" />
</p>


- Copy the path of the "bin" folder located within the wkhtmltopdf installation directory, which by default is "C:\Program Files\wkhtmltopdf\bin"

<p align="center">
<img src="./visual/ruta_de_la_carpeta_bin.png" alt="copiarrutabinwkhtmltopdf" />
</p>




- Open the Environment Variables editor. Press Windows + S, then search for "Edit the System Environment Variables" and open it
<p align="center">
<img src="./visual/abrir_editar_variables_de_entorno.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Clic on "Environment Variables"
  
<p align="center">
<img src="./visual/clic_en_environment_variables.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Edit the Path variable, either for system variables or for a specific user.
<p align="center">
<img src="./visual/edit_variable_path.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Clic on "New"
<p align="center">
<img src="./visual/Clic_on_new.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Paste the path of the "bin" folder. If the path is set to the default, it will be: "C:\Program Files\wkhtmltopdf\bin". Then click "Ok" on all subsequent windows to apply the changes.
<p align="center">
<img src="./visual/paste_binpath.png" alt="copiarrutabinwkhtmltopdf" />
</p>

## Installation Steps for Dynamic Seism:

- Install and configure the tool wkhtmltopdf. This step has been explained previously and it is recommended to review the section **"How to Install and Configure the External Tool wkhtmltopdf?"**


- Run the Dynamic Seism software as an administrator
<p align="center">
<img src="./visual/run_as_administrator_dynamicseism.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Accept the license terms and click on Next
<p align="center">
<img src="./visual/accept_the_license_terms_and_clic_next.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Select the installation folder path for the software and click on Next

<p align="center">
<img src="./visual/select_path_installation_dynamicseism.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Check the option labeled "Create a desktop shortcut" then click Next.

<p align="center">
<img src="./visual/create_icon_deskop_dynamicseism.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Verify that the installation path is correct and that an icon is created on the desktop, then click Next.
  
<p align="center">
<img src="./visual/Clic_Install_dynamicseism.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Wait while the installation proceeds

<p align="center">
<img src="./visual/wait_for_installation_dynamicseism.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- A window will appear notifying you that the software has been installed. Then, click on Finish

<p align="center">
<img src="./visual/Notification_installation_end_dynamicseism.png" alt="copiarrutabinwkhtmltopdf" />
</p>

## Main Windows

<p align="center">
<img src="./visual/principal_main_and_presentation.png" alt="copiarrutabinwkhtmltopdf" />
</p>


## How to Create a New Project?

- Click on the File menu

<p align="center">
<img src="./visual/How_Create_a_New_Project/1clic_on_file.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Click on New
<p align="center">
<img src="./visual/How_Create_a_New_Project/2clic_on_new.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Select the folder that will host the new project
<p align="center">
<img src="./visual/How_Create_a_New_Project/3select_newproject_folder.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Assign a name to the project
<p align="center">
<img src="./visual/How_Create_a_New_Project/4write_name_project.png" alt="copiarrutabinwkhtmltopdf" />
</p>

## How to load, process, and analyze the frequency content of the project's seismic records?
It is important to note that before loading the records, a new project must be created. To do this, refer to the section **"How to Create a New Project?"**

The steps to load seismic records into a project, process them by correcting the baseline or applying filters, and analyze their frequency content are as follows:

- Clic on "Load Seismic Records"

<p align="center">
<img src="./visual/How_load_process_analyze_sr/1clic_on_load_seismic_record_2.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Clic on "Select Files"

<p align="center">
<img src="./visual/How_load_process_analyze_sr/2clic_on_select_files.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Select the seismic project's seismic records
Select the project's seismic records. Keep in mind that you can choose both .txt and .AT2 files. To select .txt files, check the "text files" option, and for .AT2 files, check the "information files" option. A project in the software can include seismic records in both formats.

<p align="center">
<img src="./visual/How_load_process_analyze_sr/3select_seismic_records.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Enter the data to process and analyze the frequency content of the seismic records.
  
The data to be entered for analyzing and processing the seismic records are as follows:

**Data Presentation:** Seismic records can commonly be presented in three types: Simple Column, Multiple Column, and Time Acceleration.

**Useless Rows:** Seismic records often contain information that should be ignored to focus on acceleration data. Useless rows refer to those that the software should omit to capture the acceleration data in the seismic record. The user should enter a natural number for this.

**Acceleration Units:** Refers to the units in which accelerations are expressed in the seismic record. The user has the option to work in two units: fraction of gravity (g) and centimeters per second squared (cm/s²).

**Conversion Factor:** This factor allows the user to convert the acceleration units in the seismic record and to scale the seismic record. The user can enter a decimal number.

**Time Interval Between Acceleration Measurements (Dt):** Typically, the first lines of text in the seismic record contain information on the time interval between each acceleration measurement. The user should enter this data after reviewing these informational lines. A positive decimal number is allowed.

**Baseline Correction:** The user has the option to correct the baseline to remove a trend in the seismic record. The user can choose between Yes or No.

**Baseline Correction Polynomial Degree:** The user must enter the degree of polynomial regression for baseline correction in the seismic record. The user should enter a natural number.

**Filter Type:** Filters are applied to remove noise from the seismic record. The software uses a Butterworth filter, offering the user three options: Highpass, Lowpass, and Bandpass. There is also an option for "None" to apply no filter to the seismic record.

**Cutoff Frequency 1 and 2:** Cutoff Frequency 1 refers to the minimum frequency allowed through in the case of a bandpass filter, while Cutoff Frequency 2 represents the maximum frequency allowed with minimal attenuation. These two frequencies define a range of interest in the signal. Decimal numbers are allowed.

**Filter Order:** Refers to how quickly the filter attenuates frequencies outside a passband. The software only allows the entry of a natural number for this data entry.

**Number of Windows:** The number of windows in which the Fourier Spectrum is divided helps identify the most representative frequencies of the seismic record, which assists in determining which frequencies contain 90% of the earthquake's energy. The user should enter a natural number.

**For example 1:**

<p align="center">
<img src="./visual/How_load_process_analyze_sr/5Enter_the_data_to_process_and_analyze_the_frequency_content_of the_seismic_records_2.png" alt="copiarrutabinwkhtmltopdf" />
</p>

Once the data is entered, click on Apply

**For example 2:**

<p align="center">
<img src="./visual/How_load_process_analyze_sr/6Enter_the_data_to_process_and_analyze_the_frequency_content_of the_seismic_records.png" alt="copiarrutabinwkhtmltopdf" />
</p>



Once the data is entered, click on Apply



- After entering the data for all seismic records in the project, click OK.

<p align="center">
<img src="./visual/How_load_process_analyze_sr/7_clic_on_ok.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Wait for the software to display a window with the message: "File processing has finished," and then click OK.

<p align="center">
<img src="./visual/How_load_process_analyze_sr/8_process_analyze_results_view.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- To view the corresponding results, click on the tab "Seismic Record Processing and Signal Frequency Content Charts".

<p align="center">
<img src="./visual/How_load_process_analyze_sr/pestana_seismicrecordprocessing.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Click on the names of the records listed on the left side. With the corresponding tab active, their results will be displayed

<p align="center">
<img src="./visual/How_load_process_analyze_sr/finally_first results.png" alt="copiarrutabinwkhtmltopdf" />
</p>

## How to Calculate Dynamic Magnification?

A project must be created in the software, as outlined in the section: **How to Create a New Project?.**
Additionally, seismic records must be processed and their frequency content analyzed, as detailed in the section: **How to load, process, and analyze the frequency content of the project's seismic records?.**
Next, follow these steps:


- Click on "Calculate Dynamic Magnification"

<p align="center">
<img src="./visual/How_calculate_dynamic_magnification/01_clic_on_calculate_dynamic_magnification.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Enter the data for calculating dynamic amplification.

<p align="center">
<img src="./visual/How_calculate_dynamic_magnification/02_empty_windows.png" alt="copiarrutabinwkhtmltopdf" />
</p>

Note the following:

**Initial and Final Damping Coefficients (Zeta 1 and Zeta 2):** The software allows plotting dynamic amplification curves for various damping values. The initial damping (Zeta 1) sets the lower limit, while the final damping (Zeta 2) defines the upper limit of a range of values.

**Damping Coefficient Variation (Delta Zeta):** The user must specify an increment in the damping values to create a sequence of values, gradually ranging from the initial to the final damping.

**Period:** This refers to the period of the structure and is used to calculate dynamic amplification values based on the predominant frequency content of the seismic records and the range of damping values entered by the user.

**For Example:**
<p align="center">
<img src="./visual/How_calculate_dynamic_magnification/03_entrythedata_dynmagn.png" alt="copiarrutabinwkhtmltopdf" />
</p>
Once the data has been entered, click OK

- Wait until a window appears with the message: "File processing has finished," and then click OK.

<p align="center">
<img src="./visual/How_calculate_dynamic_magnification/04_message_dynmagn.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Click on the tab "Dynamic Magnification Charts" to view the charts

<p align="center">
<img src="./visual/How_calculate_dynamic_magnification/05_dynmagn_charts.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Click on the names of the records listed on the left side. With the corresponding tab active, the dynamic amplification results will be displayed.

<p align="center">
<img src="./visual/How_calculate_dynamic_magnification/06_finally_dynmagn.png" alt="copiarrutabinwkhtmltopdf" />
</p>

## How to generate and view the summary result graphs obtained in the software?

The graphs that summarize the results obtained in the software are generated only when at least 11 seismic records have been loaded into a project. Additionally, it is necessary to first obtain the results of the predominant frequency content of the seismic records and the results of the dynamic amplification calculation. You can review the sections **"How to Load, Process, and Analyze the Frequency Content of the Project's Seismic Records?"** and **"How to Calculate Dynamic Magnification?"** The steps to generate these graphs are described in the following paragraphs.

- Click on the "Configure Summary Settings" button.
<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/01clic_on_configure_summary_settings.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Enter the data to generate the charts that summarize the results.

<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/02Enter_the_data_in_the_Configure_Summary_Settings_window.png" alt="copiarrutabinwkhtmltopdf" />
</p>

The data the user must input to generate the charts that summarize the results obtained in the software are:

**Frequency Intervals:** Specifies the number of intervals used to group the predominant frequencies of seismic records, which will be used to generate histograms.

**Elastic Dynamic Magnification Intervals (E. Dynamic M. Intervals):** Defines the number of intervals used to group the dynamic magnification results for the elastic range, which will be represented in a histogram.

**Inelastic Dynamic Magnification Intervals (I. Dynamic M. Intervals):** Establishes the number of intervals used to classify the dynamic magnification results for the inelastic range, which will then be represented in a histogram.

**For Example:**

<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/03Example_data_in_the_Configure_Summary_Settings_window.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Wait for the notification that the chart generation has been completed.

<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/04_message_summary_charts.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Click on the "Summary Report Charts" tab.
  
<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/05_clic_on_summary_Report_Charts.png" alt="copiarrutabinwkhtmltopdf" />
</p>

## What Files and Folders Make Up a Project?

A project also consists of two folders named file and results, as well as two files called definitions_sismoanalyticsproject and _sismoanalyticsproject.
<p align="center">
<img src="./visual/project_content/1_all_files_project.png" alt="copiarrutabinwkhtmltopdf" />
</p>

The file folder contains the seismic records loaded into the project.

<p align="center">
<img src="./visual/project_content/2_files_folder.png" alt="copiarrutabinwkhtmltopdf" />
</p>

The results folder contains three subfolders named html, pdf, and xlsx. The html subfolder also contains a folder called images, where the images displayed in the graphical interface are stored, along with all the HTML files generated by the software. The pdf folder contains the results in PDF format as a report. Lastly, the xlsx folder contains the results obtained in the software.

<p align="center">
<img src="./visual/project_content/3_results_folder.png" alt="copiarrutabinwkhtmltopdf" />
</p>

The .sisproj file contains all the necessary information to reopen a project in the software. This file includes all the data entered by the user and the files loaded into the project.

## How to Open an Existing Project?

- Click on Open in the File menu

<p align="center">
<img src="./visual/How to Open an Existing Project/1_clic_on_open.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Navigate to the folder path containing an existing project and open the "_sismoanalyticsproject.sisproj"" file

<p align="center">
<img src="./visual/How to Open an Existing Project/2_open_file_sisproj.png" alt="copiarrutabinwkhtmltopdf" />
</p>



## How to Review the Reports and Results of a Project in the Software?

Locate the folder containing a project developed with the software and open the results subfolder. There, you will find:

- Reports with charts in PDF format

<p align="center">
<img src="./visual/View_Results/1_pdf_results.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Reports of the results in XLSX format

<p align="center">
<img src="./visual/View_Results/02xlsx_results.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- All the images displayed in the graphical interface

<p align="center">
<img src="./visual/View_Results/3_all_images.png" alt="copiarrutabinwkhtmltopdf" />
</p>

## Charts Displayed in the Software

- Original Record for: Acceleration, Velocity and Displacement

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/1_original_acceleration_velocity_displacement.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Comparison of Original and Corrected Records for: Acceleration, Velocity, and Displacement

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/2_comparisson_original_and_corrected_acceleration_velocity_displacement.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Corrected Record for: acceleration, velocity, and displacement

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/03_corrected_acceleration.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Filter Gain

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/04_Filter_Gain.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Arias Intensity

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/05_AriaIntensity.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Comparison between corrected acceleration and acceleration for the significant duration

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/06_Correctedacceleration_vs_accelerationforsignificantduration.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Fourier Spectrum and Frequency Content Analysis of the Entire Corrected Seismic Record

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/07_Fourier_Spectrum.png" alt="copiarrutabinwkhtmltopdf" />
</p>

<p align="center">
<img src="./visual/Charts_Displayed_in_the_Software/08_Frequency_Content_Analysis.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Dynamic Magnification Curves

<img src="./visual/Charts_Displayed_in_the_Software/9_Dynamic_Magnification_Curves.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Bar Chart generated in the Dynamic Seism software

<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/06_bar.png" alt="copiarrutabinwkhtmltopdf" />
</p>


- Combined chart of predominant frequencies: Histogram, probability distribution, Percentiles (16, 50, 84), Mean, and Standard Deviation.

<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/07_frequency_gaussian_distribution.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Combined chart of Elastic Dynamic Magnification: Histogram, probability distribution, Percentiles (16, 50, 84), Mean, and Standard Deviation.

<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/08_elastic_dynamic_magnification_gaussian_distribution.png" alt="copiarrutabinwkhtmltopdf" />
</p>

- Combined chart of Inelastic Dynamic Magnification: Histogram, probability distribution, Percentiles (16, 50, 84), Mean, and Standard Deviation.

<p align="center">
<img src="./visual/How_to_generate_and_view_the_summary_result_graphs_obtained_in_the_software/09_inelastic_dynamic_magnification_gaussian_distribution.png" alt="copiarrutabinwkhtmltopdf" />
</p>

<!--
## User Manual

You can download the User Manual by clicking on the following link:

- [Download the User Manual]()
-->

## Video for Installation

You can view and download the software installation video at the following link:
[Video](https://puceeduec-my.sharepoint.com/:v:/g/personal/meriverabo_puce_edu_ec/EQW5ooVRLFlFh4xyy14kCcYBj21V6zmJlNoDj3jcgiUeXg?e=AQzmLx&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)
