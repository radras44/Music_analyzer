import time
def execution_time (func) :
    def wrapper(*args,**kwargs) :
        start_time = time.time()
        func_return = func(*args, **kwargs)
        end_time = time.time()
        print(f"function name: {func.__name__} execution time: {end_time - start_time}")
        return func_return
    return wrapper