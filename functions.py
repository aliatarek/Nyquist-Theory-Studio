from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg
import numpy as np
from scipy.fft import rfft ,rfftfreq
import matplotlib.pyplot as plt
import pandas as pd
composedList=[]
extracted_values = []
saved_signal=0
FlagViewed=False
noisy_signal=0
composed_signal=0
fMax=0
flagnoisy=False

def updateSaved(self): #This function sends the composed_signal to the saved_signal where it is displayed
    global composed_signal
    global saved_signal
    global composedList
    global fMax
    saved_signal=composed_signal
    self.FMaxBox.display(fMax) #Updates FMax as soon as the signal is sent to the viewer
    composed_signal=0
    composedList=[]
    
def compose_signal(self):
    frequency = (self.frequencySliderComposer.value())
    self.frequencyBoxComposer.display(frequency) #Updating frequency value using slider
    
    magnitude = self.magnitudeSliderComposer.value()
    self.magnitudeBoxComposer.display(magnitude) #Updating Magnitude Value using slider
    
    phase = self.phaseSliderComposer.value()
    self.phaseBoxComposer.display(phase) #Updating Phase Value using slider
    
    snr = self.SNRSlider.value()
    self.SNRBox.display(snr) #Updating SNR Value
   
    # Generate a signal based on the slider values
    t = np.linspace(0, 1, 1000)  # Time values from 0 to 1 second (1000 values)
    signal = magnitude * np.cos(2 * np.pi * frequency * t + np.deg2rad(phase)) #Generating a sin wave with the given parameters
  
    # Plot the composed signal in Graph 1
    self.graph1Composer.clear() #Clearing the composer graph after sending to the viewer
    penk=pg.mkPen('g',width=2)
    self.plotted= self.graph1Composer.plot(t, signal, pen=penk, name='Composed Signal') 
    signaldata=[signal,frequency,phase,magnitude,snr] #This array stores all information of the current stored signal
    return signaldata  # Return the signal as a NumPy array

def add_signal(self):
    # Plot the saved signal (sum of all composed signals) in Graph 2
    t = np.linspace(0, 1, 1000)  # Time values from 0 to 1 second
    global saved_signal
    global FlagViewed
    global fMax
    global composed_signal
    if (FlagViewed): #Checks if there is already a plotted graph: resets the composed_signal
       composed_signal=0
       FlagViewed=False
       fMax=0
    composed_data=compose_signal(self) #Calls the compose function and adds the composed signal
    composed_signal += composed_data[0]  # Add the previously saved signal to itself
 
    self.graph2Composer.clear()
    penk=pg.mkPen('r',width=2)
    self.graph2Composer.plot(t, composed_signal, pen=penk, name='Saved Signal') #Printing the final composed signal
    global composedList
    composedList.append(composed_data) #Adding the newly created signal to the list  (to be able to delete)
    if composed_data[1]>fMax:
        fMax=composed_data[1]
        
    name = "Signal "+ str(len(self.signalsBoxComposer)+1) + " -  F  " + str(composed_data[1]) + "  M  " + str(composed_data[3]) + "  P  " + str(composed_data[2])
    self.signalsBoxComposer.addItem(name) #Updating the combobox with available signals
    self.signalsBoxComposer.setCurrentText(name)
        
def save(self):
    options = QFileDialog.Options() 
    file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    #Selecting a save name and location (saved as CSV)
    if file_name:
        t = np.linspace(0, 1, len(composed_signal))  # Time values from 0 to 1 second
        df = pd.DataFrame({'Time': t, 'Saved_Signal': composed_signal}) #Appends the data to the dataframe
        df.to_csv(file_name, index=False, header=False)  # Converts DF to CSV
def fHz(self):
    global fMax
    value=int(self.samplingSliderViewer.value()/fMax)
    match value: #Switch cases for updating the FMax Slider
        case 0:
            return 0
        case 1:
            return 1
        case 2:
            return 2
        case 3:
            return 3
        case 4:
            return 4
        case 5:
            return 5
        # If an exact match is not confirmed, this last case will be used if provided
        case _:
            return 5
def view_graph(self): #Main function that updates all the plots
    global saved_signal
    global fMax
    global extracted_values 
    global noisy_signal
    global flagnoisy
    global FlagViewed
    FlagViewed= True
    self.graph1Viewer.clear()
    self.graph2Viewer.clear()
    self.graph3Viewer.clear()
    #The viewer is cleared to ensure no duplication of the graphs
    self.signalsBoxComposer.clear()
    #The channelbox is also cleared to allow a new signal to be composed
    fNormalized=fHz(self) #Normalizes frequency in HZ to FMax
    num_samples = self.samplingSliderViewer.value() #Getting number of samples
    self.normalizedBoxViewer.display(fNormalized)
    self.normalizedSliderViewer.setValue(fNormalized) #Updating values in LCD
    
    step_size = len(saved_signal) / (num_samples)
    ##################################################
    if flagnoisy: #If noisy signal, it updates graphs from noisy_signal, else it updates from saved_signal
        sample_indices = [int(i) for i in np.arange(0, len(noisy_signal), step_size)] #Creating step indices for samples
        
        for item in self.graph1Viewer.items():
            if isinstance(item, pg.ScatterPlotItem):
                self.graph1Viewer.removeItem(item)# Clear the previous ScatterPlotItem if it exists
        t = np.linspace(0, 1, len(noisy_signal)) #Creating time values for the noisy signal
        penk = pg.mkPen('cyan', width=3)
       # penk=pg.mkPen('r',width=2)
        extracted_values=[]
        for i in sample_indices:
            extracted_values.append(noisy_signal[i]) #Appending the sampled data to plot it
        scatter = pg.ScatterPlotItem ( #Preparing a new scatter plot for samples
                size=10,
                pen=penk,
                brush=pg.mkBrush (0, 255, 255, 170),
               
                symbol='x'
            )
        spots = [{'pos': (t[i], saved_signal[i])} for i in sample_indices] 
        scatter.addPoints(spots)
        self.graph1Viewer.addItem(scatter) #Scatter points are added to the graph
        penk=pg.mkPen('r',width=2)
        self.graph1Viewer.plot(t, noisy_signal, pen=penk, name='Saved Signal')
        self.graph2Composer.clear()
        self.frequencyBoxViewer.display(num_samples) 
        
        return spots
    else: #This means that the signal being displayed is clear of noise, hence using the saved_signal instead of noisy_signal
        t = np.linspace(0, 1, len(saved_signal))
        sample_indices = [int(i) for i in np.arange(0, len(saved_signal), step_size)]
    
        # Clear the previous ScatterPlotItem if it exists
        for item in self.graph1Viewer.items():
            if isinstance(item, pg.ScatterPlotItem):
                self.graph1Viewer.removeItem(item)
        extracted_values=[]
        penk = pg.mkPen('cyan', width=2)
        for i in sample_indices:
            extracted_values.append(saved_signal[i])
        scatter = pg.ScatterPlotItem (
                size=12,
                pen=penk,
                brush=pg.mkBrush (0, 255, 255, 170),
                symbol='x'
            )
        spots = [{'pos': (t[i], saved_signal[i])} for i in sample_indices] 
        
        
        scatter.addPoints(spots)
        self.graph1Viewer.addItem(scatter)
        penk=pg.mkPen('r',width=2)
        self.graph1Viewer.plot(t, saved_signal, pen=penk, name='Saved Signal')
        self.graph2Composer.clear()
        self.frequencyBoxViewer.display(num_samples)
        return spots
        
def shannon_interpolation(x, y, t):
    #x period 
    #t one second 
    #y spot(position)
    interpolated_values = np.zeros_like(t, dtype=float) #Creates a zeroes array to fill with values
    delta_t = x[1] - x[0]
    
    for i, ti in enumerate(t):
        for j in range(len(x)):
            interpolated_values[i] += y[j] * np.sinc((ti - x[j]) / delta_t) #Applies Whittaker-Shannon formula

    return interpolated_values




def interpolate_graph(self):
   # global fMax
    global extracted_values
    global saved_signal
    global flagnoisy
    newtime=np.linspace(0,1,int(self.samplingSliderViewer.value()),endpoint=False) 
    if ~flagnoisy:
        t = np.linspace(0, 1, len(saved_signal))  # Time values from 0 to 1 second
     
        interpol=shannon_interpolation(newtime,extracted_values,t) #Applies 
    else:
        t=np.linspace(0,1,len(noisy_signal))
        interopl=shannon_interpolation(newtime,extracted_values,t)
   
    self.graph2Viewer.clear()
    penk=pg.mkPen('r',width=2)
    self.graph2Viewer.plot(t, interpol, pen=penk, name='Interpolated Signal')
    self.graph3Viewer.clear()
    self.graph3Viewer.plot(t,saved_signal-interpol,pen=penk,name='Difference')
  #  self.graph2Viewer.clear()


def delete_signal(self):
    global saved_signal
    global composedList
    global fMax
    global composed_signal
    t = np.linspace(0, 1, 1000)  # Time values from 0 to 1 second
    indexDeleted=self.signalsBoxComposer.currentIndex()
    self.graph2Composer.clear()
    self.signalsBoxComposer.removeItem(indexDeleted)
    newSignal=np.zeros(1000, dtype=float)
    composed_signal-=composedList[indexDeleted][0]
    composedList.pop(indexDeleted)
    fMax=0
    for i in range(len(composedList)):
        if composedList[i][1] > fMax:
            fMax= composedList[i][1]
    penk=pg.mkPen('r',width=2)
    self.graph2Composer.plot(t,composed_signal,pen=penk,name='Saved Signal')
    if len(self.signalsBoxComposer)==0:
        self.graph2Composer.clear()
    
    
def signal_noise_ratio(self):
    global saved_signal
    global noisy_signal
    global flagnoisy
    
    t = np.linspace(0, 1, len(saved_signal)) # Time values from 0 to 1 second
    current_noise = compose_signal(self)[4] #Setting the SNR value from slider 
    global flagnoisy
    if current_noise!=50:#Check if there is noise
        flagnoisy=True #if true there is noise
    else:
        flagnoisy=False #otherwise no noise
    noise_power = 10 ** (-current_noise / 10) #setting the amplitude of the noise signal according to the SNR slider
    noise = np.random.normal(0, np.sqrt(noise_power), len(saved_signal)) #creating the noise signal with the length of the original signal
    current_noise = noise #setting the new noise 
    noisy_signal = saved_signal + current_noise #adding the old signal to the new signal and saving it in another new noisy signal
    self.graph1Viewer.plot(t, noisy_signal, pen='r') #plotting the new noisy signal

def update_saved_signal(self):
        # Update the saved_signal based on the slider's value
        global saved_signal
        
        #saved_signal = np.linspace(0, self.SNRSlider.value() / 100, 1000)
        self.graph1Viewer.clear()
        t = np.linspace(0, 1, len(saved_signal))
        #self.graph1Viewer.plot(t, saved_signal, pen='red')

        # Call your signal_noise_ratio function here
        signal_noise_ratio(self)




def Load_graph(self):
    global saved_signal
    global fMax
    Signal_data = []
    Time = []

    options = QFileDialog.Options()
    file_name1, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Text Files (*.txt);;DAT Files (*.dat)", options=options)

    if file_name1:
        self.graph1Viewer.clear()
        self.graph2Viewer.clear()
        self.graph3Viewer.clear()
        
        # Determine the file extension
        file_extension = file_name1.split(".")[-1].lower()

        if file_extension in ["csv", "txt"]:
            # Read CSV or text file
            df = pd.read_csv(file_name1)
        elif file_extension == "dat":
            # Read DAT file (adjust separator as needed)
            df = pd.read_csv(file_name1, sep='\t')
        else:
            # Handle unsupported file format
            return

        Time = df.iloc[:, 0].values

        Column_length = len(df.columns) - 1
        for i in range(Column_length):
            Signal_data.append(df.iloc[:, i + 1].values)

        saved_signal = np.sum(Signal_data, axis=0)
        
        fourier=rfft(saved_signal)
        
       # plt.plot(np.abs(fourier)/(len(saved_signal)/2))
        max_freqs=[]
        
        sampling_frequency = 1 / (Time[1] - Time[0])
        sampling_period = 1 / sampling_frequency
        
        frequency=rfftfreq(len(saved_signal),sampling_period)
        normal_magnitude=np.abs(fourier)
       # pg.plot(frequency,np.abs(fourier))
        #k=0
      
        
        for i in range (len(frequency)):
         
            if frequency[i] and normal_magnitude[i]>0.0001:
                max_freqs.append(frequency[i])
           
    
                
        print(max_freqs)
        fmax=int(max(max_freqs))
        print(fmax)
        fMax=fmax
        
        self.FMaxBox.display(fMax)
        # Call the function to view the graph here (replace 'view_graph(self)' with the actual function call)
        view_graph(self)



    
def normalize(self):
    global fMax
    if self.normalizedSliderViewer.isSliderDown():
       
        self.samplingSliderViewer.setValue(self.normalizedSliderViewer.value()*fMax)
        self.normalizedBoxViewer.display(self.normalizedSliderViewer.value())
        view_graph(self)
        interpolate_graph(self)
        composed_signal=0
        composedList=[]
