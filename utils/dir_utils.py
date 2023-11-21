import os 
import shutil
import random
import zipfile
def create_control (
    path : str,
    delete_current : bool = False
    ) :
    if delete_current :   
        if (os.path.exists(path) == True):
            shutil.rmtree(path)
            
    if os.path.exists(path) == False : 
        os.mkdir(path)

def rename_dir_files (path,base_digits = 4) :
    file_paths = [os.path.join(path,file_name) for file_name in os.listdir(path)]
    for element in file_paths : print(element,"\n")
    
    dir_name = os.path.basename(path)
    
    for i,file_path in enumerate(file_paths) :
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1]
        file_number = f"{i:04}"
        new_name = f"{dir_name}_{file_number}{file_extension}" 
        new_file_path = os.path.join(path,new_name)
        print(new_file_path)
        
        os.rename(
            src=file_path,
            dst=new_file_path
        )
    
def train_test_split (main_path : str) :
    #crear directorios
    folder_names = [name for name in os.listdir(main_path) if name not in ["train","test"]]
    test_path = os.path.join(main_path,"test")
    train_path = os.path.join(main_path,"train")
    create_control(test_path, delete_current=True)
    create_control(train_path,delete_current=True)
    
    #recorrer cada directorio en main_path
    for folder_name in folder_names :
        #construir path del subdirectorio
        folder_path = os.path.join(main_path,folder_name)
        #obtener nombres de archivos del subdirectorio
        file_names =  os.listdir(folder_path)
        #obtener paths de cada archivo
        file_paths = [os.path.join(folder_path,file_name) for file_name in file_names]
        #ordenar aleatoriamente los nombres
        random.shuffle(file_paths)
  
        num_files = len(file_paths)
        num_train = int(num_files * 0.80)
        
        train_file_paths = file_paths[:num_train]
        test_file_paths = file_paths[num_train:]
        
        train_class_path = os.path.join(main_path,"train",folder_name)
        test_class_path = os.path.join(main_path,"test",folder_name)
        
        create_control(train_class_path)
        create_control(test_class_path)
        
        for file_path in test_file_paths : 
            print(f"{file_path} => {test_path}")
            shutil.copy(
                src= file_path,
                dst= test_class_path
            )
        
        for file_path in train_file_paths : 
            print(f"{file_path} => {train_path}")
            shutil.copy(
                src= file_path,
                dst= train_class_path
            )

            
       
        
    
    
    
    
    
    