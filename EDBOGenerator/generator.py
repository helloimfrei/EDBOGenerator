import pandas as pd
import os
import plotly.graph_objects as go
from edbo.plus.optimizer_botorch import EDBOplus
import EDBOGenerator.vis as vis
import datetime
import json

""" 
TO DO
work on logging feature, make it more readable and intuitive.
add programmatic result filling, don't want any interaction with csvs in run folder ideally
"""
class EDBOGenerator:
    def __init__(self,run_dir:str):
        """
        Initialize a run directory, components file, and logging file (summary.json) if not found. 
        Components file should have columns for each component, and columns named min and max with a list of objective to min max in the optimizer.
        """
        self.run_dir = run_dir
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            print('run directory did not exist, created one')
        else:
            print('run directory already exists')    
        os.chdir(self.run_dir)   
        self.run_name = f'{os.path.split(run_dir)[-1]}'

        #initialize logging file and counter for persistent logging between sessions
        if not os.path.exists(f'{self.run_name}_summary.json'):
            self.summary = {}
            with open(f'{self.run_name}_summary.json','x') as f:
                json.dump(self.summary,f)
            print('logging file (summary.json) generated in run directory')
            self._counter = 0
        else:
            with open(f'{self.run_name}_summary.json', 'r') as f:
                self.summary = json.load(f)
            self._counter = len(self.summary)    
        component_template = pd.DataFrame({
            'component_1':['a','b','',''],
            'component_2':['e','f','g','h'],
            'component_3':['i','j','k',''],
            'component_x':['m','n','o','p'],
            'min':['obj_1','obj_2','',''],
            'max':['obj_3','','','']
        })
        if not os.path.exists(run_dir + '/components.csv'):
            component_template.to_csv(run_dir + '/components.csv',index=False)
            print('components.csv file did not exist, created a template. run successfully initialized')
        else:
            print('components.csv file already exists, run successfully initialized')

    def _log_stuff(self,method:str,params:dict,result = None):
        self.summary[self._counter] = {
            'timestamp':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method':method,
            'params':params,
            'result':result
        }
        with open(f'{self.run_name}_summary.json','w') as f:
                json.dump(self.summary,f)
        self._counter+=1


    def initialize_scope(self,batch_size:int = 5, **kwargs):
        """
        Initialize the reaction scope of the optimization run, and generate a round_0 file if needed. Run this everytime you start a new session,
        even if you already have some optimization data.
        
        Parameters
        ----------
        batch_size : int
            Number of reactions EDBO will suggest each round (can alter later on, >10 may liquify your machine)
        
        Returns
        -------
        None
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
        if not os.path.exists(f'{self.run_name}_round_0.csv'):
            pd.read_csv(f'{self.run_name}_scope.csv').to_csv(f'{self.run_name}_round_0.csv',index = False)
            EDBOplus().run(objectives = self.run_objs,
                        objective_mode = self.obj_modes,
                        directory = self.run_dir,
                        filename = f'{self.run_name}_round_0.csv',
                        batch = batch_size,
                        **kwargs
                        )
            print('no round_0 file found,a new one was successfully generated. ready for data input!')
        else:
            print('round_0 file located, ready to continue optimization!')
        self._log_stuff('initialize_scope',params = {'batch_size':batch_size,'kwargs':kwargs})
    
    def input_data(self):
        """
        input round data using a pandas df (optional). you can also directly modify the generated round file
        """
        pass

    def single_round_predict(self,round_num:int = 0,batch_size:int = 5, **kwargs):
        """
        Perform one round of selection and generate required file for subsequent round.

        Parameters
        ----------
        round_num : int
            Optimization round to continue from.
        batch_size : int
            Number of reactions EDBO will suggest each round (can alter later on, >10 may liquify your machine)

        Returns
        -------
        None
        """
        EDBOplus().run(objectives = self.run_objs,
                     objective_mode = self.obj_modes,
                     directory = self.run_dir,
                     filename = f'{self.run_name}_round_{round_num}.csv',
                     batch = batch_size,
                     **kwargs
                     )
        round_result = pd.read_csv(f'pred_{self.run_name}_round_{round_num}.csv')
        next_round = round_result.filter(axis = 'columns', regex='^(?!.*(predicted_mean|predicted_variance|expected_improvement)).*')
        next_round.to_csv(f'{self.run_name}_round_{round_num+1}.csv',index = False)
        print(f'optimization successful! next round (round_{round_num+1}) is ready for data input')
        self._log_stuff('single_round_predict',params = {'round_num':round_num,'batch_size':batch_size,'kwargs':kwargs})

    def bulk_training(self,training_df:pd.DataFrame,round_num:int = 0,batch_size:int = 5,priority_limiting = False,priority_setting = False):
        """
        Updates a round file with training data from a provided training df.

        Parameters:
        -----------
        training_df : pd.DataFrame
            A dataframe containing columns for each component and columns for each objective with filled results.
        round_num : int, optional
            The round number at which you want to insert bulk training data (you can run this function at any point during your optimization).
        batch_size : int, optional
            The experimental batch size for next prediction using the functions single_round_predict or simulate_run (default is 5, beware anything higher than 10 may liquify your machine).
        priority_limiting : bool, optional
            Whether to limit training data input to rows with priority 1
            (useful when running a simulated optimization of multiple rounds).
        priority_setting : bool, optional
            Whether to set the priority of rows with newly inputted data to -1 
            (useful when running a simulated optimization of multiple rounds).

        Returns:
        --------
        None
        """
        if not os.path.exists(os.path.join(self.run_dir,'backup')):
            os.mkdir(os.path.join(self.run_dir,'backup'))
        round_input = pd.read_csv(f'{self.run_name}_round_{round_num}.csv')
        round_input.to_csv(os.path.join('backup',f'{self.run_name}_round_{round_num}_bulk_training_backup.csv'),index = False)
        to_fill = round_input.copy()
        if priority_limiting:
            to_fill = round_input[round_input.priority == 1]
        for index,row in to_fill.iterrows():
            query_list = []
            for component in self.components.keys():
                query_list.append(f'{component} == "{row[component]}"')
            query = ' & '.join(query_list)
            matching_rows = training_df.query(query)
            if not matching_rows.empty:
                results_list = matching_rows[self.run_objs].iloc[0].tolist()
                round_input.loc[index, self.run_objs] = results_list
                round_input.loc[index, 'priority'] = 1 if priority_setting else -1
                print('data from training df successfully filled for query:',query)
            else:
                print('no matching row in training df for query:',query)
            output = round_input.sort_values(by = 'priority', ascending = False)
            output.loc[0:batch_size-1,'priority'] = 1
        output.to_csv(f'{self.run_name}_round_{round_num}.csv',index = False)
        print(f'bulk data input successful! {self.run_name}_round_{round_num}.csv was overwritten and a backup was stored in {os.path.join(self.run_dir,"backup")}')
        self._log_stuff('bulk_training',params = {'training_df':training_df.to_dict(),'round_num':round_num,'batch_size':batch_size,'priority_limiting':priority_limiting,'priority_setting':priority_setting})


    def simulate_run(self,training_df:pd.DataFrame,max_rounds:int,batch_size:int=5):
        """
        Perform a simulated optimization run using bulk input data.

        Parameters
        ----------
        training_df : pd.DataFrame
            A dataframe containing columns for each component and columns for each objective with filled results.
        batch_size : int
            The number of reactions suggested by EDBO each round (beware anything higher than 10 may liquify your machine).
        max_rounds : int
            The number of rounds to simulate.

        Returns
        -------
        None
        """
        self._log_stuff('simulate_run',params = {'training_df':training_df.to_dict(),'batch_size':batch_size,'max_rounds':max_rounds})
        for round_num in range(max_rounds):
            self.bulk_training(training_df = training_df, batch_size = batch_size,round_num=round_num,priority_limiting = True, priority_setting=True)
            self.single_round_predict(round_num = round_num,batch_size = batch_size)

    def summary(self):
        """
        summarize the current state of the run as a dataframe
        """

    def visualize(self,plot_type):
        """
        visualize the prediction results of the run 
        """
        pass