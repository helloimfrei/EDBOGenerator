import pandas as pd
import os
from edbo.plus.optimizer_botorch import EDBOplus

class EDBOGenerator:
    def __init__(self,run_dir:str):
        self.run_dir = run_dir
        """
        1.initialize a run directory if not found, and also initialize components file if not found. 
        components file should have columns for each component, and columns named min and max with a list of which
        parameters to min max in the optimizer
        2. generate a dictionary from this file, assign component name:list of values and min/mix:list of components.
            make these an attribute of the instance
        3. initialize round_0 file
        """
        component_template = pd.DataFrame({
            'component_1':['a','b','c','d'],
            'component_2':['e','f','g','h'],
            'component_3':['i','j','k','l'],
            'component_x':['m','n','o','p'],
            'min':['component_1','component_2','component_3','component_x'],
            'max':['component_1','component_2','component_3','component_x']
        })
    def single_round_predict(self,round_num,batch_size):
        """
        perform one round of selection 
        """
        pass

    def bulk_training(self,training_df):
        """
        perform bulk data input from a dataframe containing columns for each component, 
        and columns for each optimizable parameter
        """

    def simulate_run(self,batch_size,round_num,max_rounds):
        """
        perform a simulated optimization run using bulk input data
        """
        pass
    def summary(self):
        """"
        summarize the current state of the run in the form of a dataframe
        """
    def visualize(self,plot_type):
        """"
        visualize the prediction results of the run 
        """