from utils import audio_utils
from utils.dev_utils import execution_time
from utils import dir_utils
from models.param_RNA import Param_RNA
import os
import json
from tqdm import tqdm
import itertools
import numpy as np
import zipfile
import multiprocessing
from tqdm import tqdm

class Process () :
    def __init__(self,test_path,model_init = True) :
        with open("process_cfg.json",mode="r") as path:
            config = json.load(path)
            
        self.config = config
        self.generator_split_mode = config["generator_split_mode"]
        self.gen_batch_split_mode = config["gen_batch_split_mode"]
        
        self.model = None
        if model_init == True : 
            self.model = Param_RNA(name=config["model_name"])
            
        self.test_path = test_path
    
    @execution_time  
    def extract_samples (self) :
        #separa un audio en partes detectando los distintos eventos (para generar data de entrenamiento)
        main_path = self.config["unprocessed_path"]
        dest_path = self.config["processed_path"]
        class_names = os.listdir(main_path)
        
        for class_name in class_names:
            count = 0
            class_path = os.path.join(main_path,class_name)
            dest_class_path = os.path.join(dest_path,class_name)
            dir_utils.create_control(class_path)
            file_names = os.listdir(class_path)
            
            for file_name in file_names :
                file_path = os.path.join(class_path,file_name) 
                signal,fs = audio_utils.read_wav(file_path)
                events = audio_utils.event_split(signal,fs)
                
                for event in events :
                    count += 1
                    fragment_number = f"{count:05}"
                    fragment_name = f"{class_name}_{fragment_number}.wav"   
                   
                    dir_utils.create_control(dest_class_path)
                    
                    fragment_path = os.path.join(dest_class_path,fragment_name)   
                    print(fragment_path)
                    audio_utils.write_wav(
                        signal=event,
                        fs=fs,
                        path=fragment_path
                    )   
    
    @execution_time
    def normalize_audio_length (self,time = 1.5) :
    #hace que todos los audios duren lo mismo y tengan la misma tasa de muestreo
        main_path = self.config["processed_path"]
        class_names = os.listdir(main_path)
        
        for class_name in tqdm(class_names) : 
            if class_name != "train" and class_name != "test" :  
                class_path = os.path.join(main_path,class_name)
                file_names = os.listdir(class_path)
                
                for file_name in file_names:
                    file_path = os.path.join(class_path,file_name)
                    
                    audio_utils.time_normalizer(path=file_path,write=True,time=time)
        
    @execution_time
    def exchange_data (self,max_audio_per_folder = None,max_comb_audios : int = 3,comprobation = False) : 
    #hace combinatoria entre conjuntos de paths de audios, desde 1 elemento hasta max_comb_audios
    #las combinaciones son guardadas en un zip como cadenas de bytes
    #lo que se guarda no es el audio sino algunos parametros (caracteristicas de frecuencia/nota y amplitud)
    
    #variables iniciales
        param_generator = audio_utils.Param_generator()
        main_path = self.config["processed_path"]
        dest_path = self.config["exchanged_path"]
        loaded_audios = {}
        folder_names = os.listdir(main_path)
        folder_paths = [os.path.join(main_path,folder_name) for folder_name in folder_names]
        
        #---------obtener todas las posibles combinaciones segun los argumentos del metodo----------
        #cada combinacion es una clase (clase en el contexto de inteligencia artificial)
        all_folder_combs = []
        for i in range(1,max_comb_audios + 1) : 
            folder_combs = list(itertools.combinations(folder_paths,i))
            all_folder_combs.extend(folder_combs)
            print(f"combinations with max {i} notes = {len(folder_combs)}")
            
        #determinar numero de muestras por clases (segun el argumento max_audio_per_folder)
        min_folder_file_num = min([len(os.listdir(folder_path)) for folder_path in folder_paths])
        print("min_folder_file_num:",min_folder_file_num)
        if max_audio_per_folder is None : 
            max_comb_samples= min_folder_file_num
        elif max_audio_per_folder <= min_folder_file_num :
            max_comb_samples = max_audio_per_folder
        else :
            max_comb_samples = min_folder_file_num
        
        #realizar combinatoria (como paths)
        all_file_comb_sets = []    
        for folder_comb in all_folder_combs : 
            file_groups = []
            for folder_path in folder_comb : 
                file_names = os.listdir(folder_path)
                if(max_audio_per_folder is not None) : 
                    file_names = file_names[:max_audio_per_folder]
                    
                file_paths = [os.path.join(folder_path,file_name) for file_name in file_names]
                file_groups.append(file_paths)
                
            file_combs = list(itertools.product(*file_groups))[:int(max_comb_samples)]
            all_file_comb_sets.append(file_combs)

        print(f"total file combinations : {len(np.array(all_file_comb_sets).reshape(-1))}")
        print(f"generating...")
        
        combinations_data = []
        
        #convertir combinaciones en audio
        for file_comb_set in tqdm(all_file_comb_sets):
            notes = [os.path.basename(path).split("_")[0] for path in file_comb_set[0]]
            comb_name = "-".join(notes)
            
            for i,file_comb in enumerate(file_comb_set):
                file_name = f"{comb_name}_{i:04}.bin"
                bin_file_path = os.path.join(comb_name,file_name)
                signal_set = []
                for wav_file_path in file_comb : 
                    if wav_file_path not in loaded_audios :
                        signal,fs = audio_utils.read_wav(path=wav_file_path)
                        loaded_audios[wav_file_path] = (signal,fs)
                    else : 
                        signal,fs = loaded_audios[wav_file_path]
                
                    signal_set.append((signal,fs))
                    
                combined_file_signal,combined_file_fs = audio_utils.combine(signal_set)
                
                combinations_data.append(((combined_file_signal,combined_file_fs),bin_file_path))
                
        #convertir combinaciones de audios en parametros de frecuenciaNota/magnitud (normalizado a valores entre 0 y 1)
        process_num = 8
        pool = multiprocessing.Pool(processes=process_num)
        combs_signal_data = [data[0] for data in combinations_data]
        chunk_len = len(combs_signal_data) // process_num
        combs_signal_data_chunks = [combs_signal_data[i : i + chunk_len] for i in range(0,len(combs_signal_data),chunk_len)]
        comb_paths = [data[1] for data in combinations_data]
        
        #aqui se obtienen los parametros
        chunk_result_catcher = pool.map_async(param_generator.get_freq_param_set,combs_signal_data_chunks)
        chunk_results = chunk_result_catcher.get()
        comb_params = []
        for chunk_result in chunk_results : 
            comb_params.extend(chunk_result)
        
        #unir cada comb_param con con el path de su muestra de clase correspondiente
        #posteriormente guardarlo dentro de un zip como cadena de bytes
        combinations_data = [(comb_params[i],comb_paths[i]) for i in range(len(combinations_data))]
        
        zip_path = os.path.join(dest_path,"exchanges.zip")
        with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zipf:
            print("escribiendo y comprimiendo en",zip_path)
            for i in tqdm(range(len(combinations_data))):
                params = combinations_data[i][0]
                path = combinations_data[i][1]
                if comprobation == True : 
                    max_value = np.argmax(params[:,1])
                    print(path,"===> :")
                    print(params[max_value])
                
                with zipf.open(path,"w") as f : 
                    f_data = np.array(params,dtype=np.float32)
                    f.write(f_data.tobytes())      

    @execution_time
    def preprocessing(self) : 
        if self.model is not None :     
            self.model.preprocessing()
        
    @execution_time
    def train(self) : 
        if self.model is not None :     
            self.model.train()
        
    @execution_time
    def test(self,sample_time = None) :
        if self.model is not None :     
            file_names : list = os.listdir(self.test_path)
            sorted_file_names = sorted(file_names,key=lambda name : len(name))
            file_paths = [os.path.join(self.test_path,name) for name in sorted_file_names]
            
            for path in file_paths :        
                self.model.test(path,sample_time=sample_time)
                
if __name__ == "__main__" : 
    process = Process(os.path.join("datasets","test_audios"),model_init = True)
    process.extract_samples()
    process.normalize_audio_length(time=0.3)
    process.exchange_data(
        max_folder_elements=None,
        max_comb_elements=3
    )
    process.preprocessing()
    process.train()
    
   
  
    
    
