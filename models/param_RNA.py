import json
from utils.note_model import Note_model
import keras
from keras.utils import to_categorical
import os
from utils import audio_utils
from sklearn import preprocessing
from tqdm import tqdm
import keras.regularizers as regularizers
import numpy as np
import joblib
import sys
import keras.layers as layers
import keras.callbacks as callbacks
class Param_RNA (Note_model) : 
    def __init__ (self,name) : 
        with open("process_cfg.json",mode="r") as path:
            self.config = json.load(path) 
        self.model_path = os.path.join(self.config["trained_models_path"],f"{name}.h5")
        self.encoder_path = os.path.join(self.config["trained_models_path"],f"{name}.pkl")
        self.current_model,self.current_encoder = super().load(self.model_path,self.encoder_path)
        self.x_train = None
        self.y_train = None
        self.classes = None
        self.input_shape = None
        self.dropout_rate = 0.2
        self.num_classes = None
    def preprocessing (self) : 
        print("cargando data...")
        x_train,y_train,freqs = super().load_exchanges(
            path=f"{self.config['exchanged_path']}/exchanges.zip"
        )
        self.input_shape = x_train[0].shape
        
        #formatear etiquetas
        label_encoder = preprocessing.LabelEncoder()
        encoded_y_train = label_encoder.fit_transform(y_train)
        self.current_encoder = label_encoder
        self.num_classes = len(label_encoder.classes_)
        joblib.dump(label_encoder,self.encoder_path)
        
        self.x_train,self.y_train = x_train,encoded_y_train    
        
        # for i in range(0,len(y_train),1) : 
        #     print("y_train:",y_train[i])
        #     print("encoded_y_train:",encoded_y_train[i])
        #     print("class by encoded_y_train",label_encoder.classes_[encoded_y_train[i]])
        #     data = np.column_stack((freqs,x_train[i].reshape(-1)))
        #     print(data)
       
        print("num_classes =",self.num_classes)
        print("input_shape:,",self.input_shape) 
        print(f"x_train:",self.x_train.shape)
        print(f"encoded y_train:",self.y_train.shape)
            
        return self
            
    def train (self) : 
        model = keras.Sequential()
        # Capa de entrada
        model.add(layers.Input(shape=(97, 1)))
        # Capa oculta
        model.add(layers.Dense(units=64, activation="tanh"))
        model.add(layers.Dense(units=128, activation="tanh"))
        model.add(layers.Dense(units=64, activation="tanh"))
        model.add(layers.Dense(units=32, activation="relu"))
        # Capa de salida
        model.add(layers.Flatten())
        model.add(layers.Dense(units=self.num_classes, activation="softmax"))
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        
        confirmation = input("confirmar para comenzar el entrenamiento(y/n):")
        if(confirmation.lower() == "n") : 
            sys.exit()
            
        callback_set = [
            callbacks.EarlyStopping(
                monitor="accuracy",
                patience=4,
                verbose=1,
                restore_best_weights=True
            )
        ]
            
        model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=10,
            batch_size=16,
            validation_split=0.2,
            callbacks=callback_set
        )
        
        model.save(self.model_path)
        self.current_model = model
        
    def test (self,path,sample_time = None) : 
        
        with open("assets/notes.json") as json_f : 
            notes_dict : dict = json.load(json_f)["note_freqs"]
            notes_arr = list(notes_dict.keys())
            print(notes_arr)
        
        if self.current_model is not None : 
            #realizar prediccion
            model : keras.models.Sequential = self.current_model 
            mag_param_batch,max_sample_amps = super().mag_param_batch_generator(
                path=path,
                sample_time=sample_time,
                return_max_sample_amps=True
                )
            
            batch_predictions_set = model.predict(mag_param_batch)
            
            results = []
            print(f"=======> archivo: {os.path.basename(path)}")
            for i,predictions in enumerate(batch_predictions_set) : 
                # obtener las 5 clases mas probables
                bests = np.argsort(predictions)[-5:]
                probs = [round(predictions[i] ,2) for i in bests]
                class_names = self.current_encoder.inverse_transform(bests)
                
                #determinar amplitud maxima del segmento de audio de la prediccion
                max_amp = max_sample_amps[i]
                
                #obtener prediccion final segun ciertos criterios como la probabilidad y la amplitud
                final_predicted_class = self.resolve_note_prediction(class_names=class_names,probs=probs,max_amp=max_amp)
                print(final_predicted_class,max_amp)
                
                #obtener posiciones de cada nota
                positions = []
                if final_predicted_class != "-" : 
                    for note in final_predicted_class.split("-") :
                        positions.append(notes_arr.index(note))
                    
                #establecer sample time si es None, para determinar inicio y final del audio y evitar un error
                if sample_time is None : 
                    temp_signal,temp_fs = audio_utils.read_wav(path=path)    
                    duration = len(temp_signal) / temp_fs
                    sample_time = duration
                    
                #crear y almacenar diccionario con propiedades sobre la prediccion final
                result = {
                    "notes" : final_predicted_class,
                    "positions" : positions,
                    "start_time" : round(sample_time * i,1),
                    "end_time" : round((sample_time * i) + sample_time,1)
                }
                
                results.append(result)
                    
            return results
            
            
        
    
      
        

        
        