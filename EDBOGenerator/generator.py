import pandas as pd
import os
from edbo.plus.optimizer_botorch import EDBOplus

class EDBOGenerator:
    def __init__(self,run_dir:str):
        self.run_dir = run_dir
        self.run_name = f'{os.path.split(run_dir)[-1]}'
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
                                         filename=f'{self.run_name}_scope.csv',
                                         check_overwrite = False
                                         )
        print('scope successfully initialized')
        pd.read_csv(f'{self.run_name}_scope.csv').to_csv(f'{self.run_name}_round_0.csv',index = False)
        EDBOplus().run(objectives = self.run_objs,
                     objective_mode = self.obj_modes,
                     directory = self.run_dir,
                     filename = f'{self.run_name}_round_0.csv',
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
                     filename = f'{self.run_name}_round_{round_num}.csv',
                     batch = batch_size
                     )
        round_result = pd.read_csv(f'pred_{self.run_name}_round_{round_num}.csv')
        next_round = round_result.filter(axis = 'columns', regex='^(?!.*(predicted_mean|predicted_variance|expected_improvement)).*')
        next_round.to_csv(f'{self.run_name}_round_{round_num+1}.csv',index = False)
        print(f'optimization successful! next round (round_{round_num+1}) is ready for data input')

    def bulk_training(self,training_df:pd.DataFrame,round_num:int = 0,batch_size:int = 5) -> dict:
        """
        1. take a dataframe with columns for each component, and columns for each objective with filled results
        2. search round file for each component combination using query strings
        3. input results from training df into objective columns in round file, set priority to -1
        4. sort final df by priority
        """
        if not os.path.exists(os.path.join(self.run_dir,'backup')):
            os.mkdir(os.path.join(self.run_dir,'backup'))
        round_input = pd.read_csv(f'{self.run_name}_round_{round_num}.csv')
        round_input.to_csv(os.path.join('backup',f'{self.run_name}_round_{round_num}_bulk_training_backup.csv'),index = False)
        for index,row in training_df.iterrows():
            results_list = [row[objective] for objective in self.run_objs]
            query_list = []
            for component in self.components.keys():
                query_list.append(f'{component} == "{row[component]}"')
            query = ' & '.join(query_list)
            round_input.loc[round_input.query(query).index[0],self.run_objs] = results_list
            round_input.loc[round_input.query(query).index[0],'priority'] = -1
            output = round_input.sort_values(by = 'priority', ascending = False)
            output.loc[0:batch_size-1,'priority'] = 1
        output.to_csv(f'{self.run_name}_round_{round_num}.csv',index = False)
        print(f'bulk data input successful! {self.run_name}_round_{round_num}.csv was overwritten and a backup was stored in {os.path.join(self.run_dir,"backup")}')


    def simulate_run(self,training_df,batch_size,round_num,max_rounds):
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