import pandas as pd
import os
from edbo.plus.optimizer_botorch import EDBOplus

class EDBOGenerator:
    def __init__(self,run_dir:str):
        self.run_dir = run_dir
        os.chdir(self.run_dir)
        """
        1.initialize a run directory if not found, and also initialize components file if not found. 
        components file should have columns for each component, and columns named min and max with a list of objective to min max in the optimizer
        """
        component_template = pd.DataFrame({
            'component_1':['a','b','',''],
            'component_2':['e','f','g','h'],
            'component_3':['i','j','k',''],
            'component_x':['m','n','o','p'],
            'min':['obj_1','obj_2','',''],
            'max':['obj_3','','','']
        })
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            print('run directory did not exist, created one')
        else:
            print('run directory already exists')
        if not os.path.exists(run_dir + '/components.csv'):
            component_template.to_csv(run_dir + '/components.csv',index=False)
            print('components.csv file did not exist, created a template. run successfully initialized')
        else:
            print('components.csv file already exists, run successfully initialized')
    
    def initialize_scope(self,batch_size:int = 5):
        """
        initialize the scope of the optimization run, generate a round_0 file
        """
        component_frame = pd.read_csv(self.run_dir + '/components.csv')
        self.components = {k:component_frame[k].dropna().tolist() for k in component_frame.drop(columns = ['min','max']).columns}
        obj_list = {'min':component_frame['min'].dropna().to_list(), 'max':component_frame['max'].dropna().to_list()}
        self.run_objs = [obj for sublist in obj_list.values() for obj in sublist]
        self.obj_modes = [mode for mode in obj_list.keys() for obj in obj_list[mode]]
        EDBOplus().generate_reaction_scope(components=self.components,
                                         directory=self.run_dir,
                                         filename=f'{os.path.split(self.run_dir)[-1]}_scope.csv',
                                         check_overwrite = False
                                         )
        print('scope successfully initialized')
        pd.read_csv(f'{os.path.split(self.run_dir)[-1]}_scope.csv').to_csv(f'{os.path.split(self.run_dir)[-1]}_round_0.csv',index = False)
        EDBOplus().run(objectives = self.run_objs,
                     objective_mode = self.obj_modes,
                     directory = self.run_dir,
                     filename = f'{os.path.split(self.run_dir)[-1]}_round_0.csv',
                     batch = batch_size
                     )
        print('round_0 file successfully generated, ready for data input!')
    
    def input_data(self):
        """
        input round data using a pandas df (optional). you can also directly modify the generated round file
        """
        pass

    def single_round_predict(self,round_num:int = 0,batch_size:int = 5):
        """
        perform one round of selection and generate required file for subsequent round
        """
        EDBOplus().run(objectives = self.run_objs,
                     objective_mode = self.obj_modes,
                     directory = self.run_dir,
                     filename = f'{os.path.split(self.run_dir)[-1]}_round_{round_num}.csv',
                     batch = batch_size
                     )
        round_result = pd.read_csv(f'pred_{os.path.split(self.run_dir)[-1]}_round_{round_num}.csv')
        next_round = round_result.filter(axis = 'columns', regex='^(?!.*(predicted_mean|predicted_variance|expected_improvement)).*')
        next_round.to_csv(f'{os.path.split(self.run_dir)[-1]}_round_{round_num+1}.csv',index = False)
        print('optimization successful! next round is ready for data input')

    def bulk_training(self,training_df):
        """
        perform bulk data input from a dataframe containing columns for each component, 
        and columns for each run objective with filled results
        """
        pass

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
        pass