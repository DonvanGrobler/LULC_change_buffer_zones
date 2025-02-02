# **LULC Change Buffer Zones**  
*A Land Use and Land Cover Change (LULCC) Analysis Tool for South African National Parks' Buffer Zones*  

## 📌 **Overview**  
This repository contains the code and methodology used for analyzing **Land Use and Land Cover Change (LULCC)** in the buffer zones of **South African National Parks (SANParks)**.  

The tool is developed as part of a **Master’s thesis** and aims to **facilitate the monitoring and evaluation** of LULC- and associated environmental changes around protected areas using **Google Earth Engine (GEE) and Dynamic World LULC dataset**.  

### **Key Objectives**  
- Characterizing **primary LULC trends** in buffer zones.  
- Performing **spatio-temporal analyses** of LULC change.  
- Using **intensity analysis** to quantify LULC changes over time.  
- Visualizing LULC transitions using **Sankey diagrams** and **heatmaps**.  
- Encouraging adoption of this tool by **Parks and Conservation Authorities**.  

---

## 📖 **Background**  
Protected areas (PAs) in South Africa (and other parts of the world) face challenges due to **anthropogenic activities** within their buffer zones. Therefore, understanding **LULC changes** in these zones is crucial for **biodiversity conservation, land management, and policy-making**.  

This project provides a **replicable workflow** to assess LULC changes and their impact on national parks, enabling stakeholders such as:  

- 🏞 **Park Management (e.g., SANParks)**  
- 🌍 **Environmental Scientists**  
- 🏛 **Conservation Policy Makers**  
- 📡 **GIS & Remote Sensing Experts**  

---

## 📂 **Repository Structure**  
```
LULC_change_buffer_zones/  
│── data/                 # LULC datasets for different parks  
│── notebooks/            # Jupyter Notebooks for analysis  
│── src/                  # Python scripts for analysis and visualization  
│── results/              # Output plots, maps, and tables  
│── .gitignore            # Files to exclude from Git  
│── LICENSE               # License information  
│── README.md             # Project documentation  
│── requirements.txt      # Dependencies  
```

---

## 🔧 **Installation & Setup**
To run the Jupyter Notebook, follow these steps:

### **1️⃣ Clone the Repository**
```
git clone https://github.com/YOUR_USERNAME/LULC_change_buffer_zones.git
cd LULC_change_buffer_zones
```
### **2️⃣ Set Up the Environment**
Install required dependencies:
```pip install -r requirements.txt```
Ensure you have:
- Python 3.8+
- Jupyter Notebook
- Google Earth Engine (GEE) API
- geemap
- pandas, numpy, matplotlib, seaborn, plotly
### **3️⃣ Authenticate Google Earth Engine (GEE)***
```
import ee  
ee.Authenticate()  
ee.Initialize()
```
### **4️⃣ Run the Jupyter Notebook**
```
jupyter notebook notebooks/LULC_analysis.ipynb
```

---

## 📊 **Features**
✔ Dynamically loads LULC datasets for selected parks  
✔ Visualizes changes through line graphs, Sankey diagrams, and heatmaps  
✔ Computes intensity metrics (Time, Category, Transition Intensity)  
✔ Interactive selection widgets for filtering specific transitions  
✔ Geospatial mapping of filtered LULCC transitions  

---

## 🗺️ **Example Outputs**
1️⃣ **LULC Trends Over Time**  
*Line graph comparing changes in different land cover classes in a park’s buffer zone.*

2️⃣ **Sankey Diagram for LULC Transitions**  
*Illustrates the flow of land cover transitions between years.*

3️⃣ **Heatmaps of Transition Intensity**  
*Visualizes which land cover categories gained or lost area in each significant interval.*

4️⃣ **Filtered Transition Maps**  
*Displays areas where specific LULC transitions occurred using Google Earth Engine.*

---

## 📈 **Future Improvements**
🔹 Expand datasets to include additional years and AoIs.  
🔹 Enhance automation via GitHub Actions for data updates.  
🔹 Improve UI/UX for interactive analysis.  

---

## 📜 **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## 👨‍💻 **Contributors**
Donvan Grobler (Lead Researcher & Developer)  
Prof. D.P. Cilliers (Supervisor)  
Potential Collaborators Welcome!  

---

## 💬 **Feedback & Collaboration**
If you are interested in collaborating, improving, or applying this tool, feel free to: 📧 Email: donvangrobler@gmail.com  


---

## 📚 **Data Sources & Libraries**
This project relies on several open datasets and libraries for analysis:

- 🌍 **Dynamic World LULC Dataset (Google Earth Engine)**  
*Paper: Brown et al. (2022). Dynamic World, near real-time global 10 m land use land cover mapping*
- 🛰 **Google Earth Engine (GEE)**  
*Used for geospatial data processing & visualization.*
- 🗺 **geemap (Python Library)**  
*Python wrapper for Google Earth Engine, used for mapping & analysis.*
- 📊 **Python Libraries Used:**  
`pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly` → for data analysis & visualization.  
`ipywidgets` → for interactive widgets in Jupyter Notebooks.
