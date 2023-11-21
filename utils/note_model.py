import numpy as np
from utils import audio_utils
import os
from tqdm import tqdm
from sklearn import preprocessing
from keras.models import load_model
import zipfile
import joblib
                    
class Note_model () : 
    def load_exchanges (self,path) : 
        x = []
        y = []
        z = None
        with zipfile.ZipFile(path,"r") as zipf:
            file_pahts = zipf.namelist()  
            
            for file_path in file_pahts : 
             
                with zipf.open(file_path,"r") as f : 
                    file_class = file_path.split("/")[0]
                    bins = f.read()
                    np.set_printoptions(suppress=True)  # Ajustar opciones de impresión
                    data = np.frombuffer(bins,dtype=np.float32).reshape(-1,2)
                    
                    if z is None : z = data[:,0]
                
                    x.append(data[:,1])
                    y.append(file_class)
                        
        return np.expand_dims(np.array(x),axis=-1),np.array(y),np.array(z)
    
    def mag_param_batch_generator(self,path,sample_time = None,return_max_sample_amps = False) : 
        param_generator = audio_utils.Param_generator()
        signal,fs = audio_utils.read_wav(path)
        max_audio_amp = np.max(np.abs(signal))
        
        #obtener set de paramatros de cada segmento, en caso de que sea None solo sera un segmento
        if sample_time is not None : 
            samples = audio_utils.split(signal,fs,sample_time)
            sample_set = [(sample,fs) for sample in samples]
            param_sets,max_sample_amps = param_generator.get_freq_param_set(
                sample_set=sample_set,
                print_progress=True,
                return_max_sample_amps=True
            )
        else :
            param_sets,max_sample_amps = param_generator.get_freq_param_set(
                sample_set=[(signal,fs)],
                print_progress=False,
                return_max_sample_amps=True
            )
            
        #eliminar frecuencias y expandir dimensiones para crear un lote
        mag_param_set = [params[:,1] for params in param_sets]
        mag_param_batch = np.expand_dims(np.array(mag_param_set),axis=-1)
        
        if return_max_sample_amps == True : 
            norm_max_sample_amps = [mag / max_audio_amp for mag in max_sample_amps]            
            return mag_param_batch,norm_max_sample_amps
        
        if return_max_sample_amps == False : 
            return mag_param_batch
        
    def class_encoder (self,y_train) :
        encoder = preprocessing.LabelEncoder()
        
        encoder.fit(y_train)
        return encoder

    def load (self,model_path,encoder_path) : 
        if os.path.exists(model_path) and os.path.exists(encoder_path) : 
            model = load_model(model_path)
            encoder = joblib.load(encoder_path)
            return model,encoder
        else :
            return None,None
        
    def resolve_note_prediction (self,class_names,probs,max_amp) :
        result_class = "-"
        allowed = False
        
        #amplitude critaria
        #1.1
        if(max_amp < 0.1) : 
            return result_class
        
        #probabilty critaria  
        #2.1
        if(probs[-1] > 0.9) :
            result_class = class_names[-1]   
            return result_class
        
        #2.2
        if(probs[-1] > 0.5 and (probs[-1] - probs[-2]) < 0.4) : 
            class_1 : str = class_names[-1].split("-")
            class_2 : str = class_names[-2].split("-")
            notes = set(class_1 + class_2)
            new_class = "-".join(notes)
            result_class = new_class
            return result_class

        return result_class
        
        

    
     
    
            
            
       
            

        
        
     
        
        
        
     
        
   

            
            
                        
         
        