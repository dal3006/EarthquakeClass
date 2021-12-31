# EarthquakeClass
Seismic Event Classification using Convolutional Neutral Network applied to time series data

This machine learning application to seismology uses a CNN to classify seismic events into earthquakes, active seismic sources and noise based on raw waveform records with a duration of 100 seconds.

Package Requirements:
The pre-processing of the seismic data requires the Obspy package which is easy to install with conda.
Keras
tensorflow

# Tutorial
1) Download the code and data. Create a new folder in the cwd called: "data" and unpack seismic_data.zip within that folder.
2) Run: 4_save_spec_as_matrix.py - This script performs the complete preprocessing steps including deterning, computing spectrograms, generating the feature matrix and labels which are all saved as a compact MatLab binary (labquake_spec.mat)
Set: dPar['showPlot'] = True, to see some examplary plots of waveform spectrograms.
4) Run: 6_spec_class_CNN.py - This script performs training and model performance evalutions based on validation and testing data. The code uses the Keras API and tensorflow. 
 
The ANN architecture includes the following laters: Conv2D, MaxPool, Conv2D, MaxPool, FC(relu), DropOut, FC(softmax)
The code generates plots of loss function, validation curve and some examples of incorrect predictions.
The overall accuracy for the provided data is 99%, meaning that earthquakes are easily detectable based on the energy signatures in the frequency-time domain.
