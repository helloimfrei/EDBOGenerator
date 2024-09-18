import os
import json
import datetime
import pandas as pd
from .vis import scatter
from edbo.plus.optimizer_botorch import EDBOplus

class EDBOGenerator:
    def __init__(self, run_dir: str):
        """
        Initialize a run directory, components file, and logging file (summary.json) if not found.
        Components file should have columns for each component, and columns named min and max with a list of objective to min max in the optimizer.
        """
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(os.getcwd(), run_dir)
        self.run_dir = run_dir
        self.run_name = os.path.basename(run_dir)
        self.summary = {}
        self._counter = 0
        print(f'run directory {"created" if not os.path.exists(run_dir) else "already exists"}')
        os.makedirs(run_dir, exist_ok=True)
        os.chdir(self.run_dir)
        self._initialize_logging()
        self._initialize_components()

    def _initialize_logging(self):
        summary_path = os.path.join(self.run_dir, f'{self.run_name}_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
            self._counter = len(self.summary)
        else:
            with open(summary_path, 'x') as f:
                json.dump(self.summary, f)
            print('logging file (summary.json) generated in run directory')

    def _initialize_components(self):
        component_template = pd.DataFrame({
            'component_1': ['a', 'b', '', ''],
            'component_2': ['e', 'f', 'g', 'h'],
            'component_3': ['i', 'j', 'k', ''],
            'component_x': ['m', 'n', 'o', 'p'],
            'min': ['obj_1', 'obj_2', '', ''],
            'max': ['obj_3', '', '', '']
        })

        components_path = os.path.join(self.run_dir, 'components.csv')
        if not os.path.exists(components_path):
            component_template.to_csv(components_path, index=False)
            print('components.csv file did not exist, created a template. Run successfully initialized')
        else:
            print('components.csv file already exists, run successfully initialized')

    def _log_stuff(self, method: str, params: dict, result=None):
        self.summary[self._counter] = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': method,
            'params': params,
            'result': result
        }
        summary_path = os.path.join(self.run_dir, f'{self.run_name}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f)
        self._counter += 1

    def initialize_scope(self, batch_size: int = 5, **kwargs):
        """
        Initialize the reaction scope of the optimization run, and generate a round_0 file if needed. Run this everytime a new session is started,
        even if some optimization data has already been collected.
        
        Parameters
        ----------
        batch_size : int
            Number of reactions EDBO will suggest each round (can alter later on, >10 may liquify your machine)
        
        Returns
        -------
        None
        """
        component_frame = pd.read_csv(os.path.join(self.run_dir, 'components.csv'))
        self.components = {col: component_frame[col].dropna().tolist() for col in component_frame.columns if col not in ['min', 'max']}
        obj_list = {'min': component_frame['min'].dropna().tolist(), 'max': component_frame['max'].dropna().tolist()}
        self.run_objs = [obj for sublist in obj_list.values() for obj in sublist]
        self.obj_modes = [mode for mode in obj_list.keys() for obj in obj_list[mode]]

        EDBOplus().generate_reaction_scope(
            components=self.components,
            directory=self.run_dir,
            filename=f'{self.run_name}_scope.csv',
            check_overwrite=False
        )
        print('scope successfully initialized')

        round_0_path = os.path.join(self.run_dir, f'{self.run_name}_round_0.csv')
        if not os.path.exists(round_0_path):
            pd.read_csv(f'{self.run_name}_scope.csv').to_csv(round_0_path, index=False)
            EDBOplus().run(
                objectives=self.run_objs,
                objective_mode=self.obj_modes,
                directory=self.run_dir,
                filename=round_0_path,
                batch=batch_size,
                **kwargs
            )
            print('No round_0 file found; a new one was successfully generated. Ready for data input!')
        else:
            print('round_0 file located, ready to continue optimization!')

        self._log_stuff('initialize_scope', params={'batch_size': batch_size, **kwargs})

    def input_data(self):
        """
        Input round data programmatically rather than directly in the csv.
        """
        pass

    def single_round_predict(self, round_num: int = 0, batch_size: int = 5, **kwargs):
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
        EDBOplus().run(
            objectives=self.run_objs,
            objective_mode=self.obj_modes,
            directory=self.run_dir,
            filename=f'{self.run_name}_round_{round_num}.csv',
            batch=batch_size,
            **kwargs
        )

        round_result = pd.read_csv(f'pred_{self.run_name}_round_{round_num}.csv')
        next_round = round_result.loc[:, ~round_result.columns.str.contains(r'predicted_mean|predicted_variance|expected_improvement')]
        next_round.to_csv(os.path.join(self.run_dir, f'{self.run_name}_round_{round_num + 1}.csv'), index=False)

        print(f'Optimization successful! Next round (round_{round_num + 1}) is ready for data input')
        self._log_stuff('single_round_predict', params={'round_num': round_num, 'batch_size': batch_size, **kwargs})

    def bulk_training(self, training_df: pd.DataFrame, round_num: int = 0, batch_size: int = 5,
                      priority_limiting: bool = False, priority_setting: bool = False):
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
            Internal use: Whether to limit training data input to rows with priority 1
            (useful when running a simulated optimization of multiple rounds).
        priority_setting : bool, optional
            Internal use: Whether to set the priority of rows with newly inputted data to -1 
            (useful when running a simulated optimization of multiple rounds).

        Returns:
        --------
        None
        """
        backup_dir = os.path.join(self.run_dir, 'backup')
        os.makedirs(backup_dir, exist_ok=True)

        round_file = os.path.join(self.run_dir, f'{self.run_name}_round_{round_num}.csv')
        backup_file = os.path.join(backup_dir, f'{self.run_name}_round_{round_num}_bulk_training_backup.csv')

        round_input = pd.read_csv(round_file)
        round_input.to_csv(backup_file, index=False)

        to_fill = round_input[round_input.priority == 1] if priority_limiting else round_input.copy()

        for index, row in to_fill.iterrows():
            query = ' & '.join([f'{component} == "{row[component]}"' for component in self.components.keys()])
            matching_rows = training_df.query(query)

            if not matching_rows.empty:
                round_input.loc[index, self.run_objs] = matching_rows[self.run_objs].iloc[0].tolist()
                round_input.loc[index, 'priority'] = -1 if priority_setting else 1
                print(f'Data from training df successfully filled for query: {query}')
            else:
                print(f'No matching row in training df for query: {query}')

        round_input.sort_values(by='priority', ascending=False, inplace=True)
        round_input.iloc[:batch_size, round_input.columns.get_loc('priority')] = 1

        round_input.to_csv(round_file, index=False)
        print(f'Bulk data input successful! {round_file} was overwritten and a backup was stored in {backup_dir}')
        self._log_stuff('bulk_training', params={
            'training_df': training_df.to_dict(),
            'round_num': round_num,
            'batch_size': batch_size,
            'priority_limiting': priority_limiting,
            'priority_setting': priority_setting
        })

    def simulate_run(self, training_df: pd.DataFrame, max_rounds: int, batch_size: int = 5):
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
        self._log_stuff('simulate_run', params={'training_df': training_df.to_dict(), 'batch_size': batch_size, 'max_rounds': max_rounds})

        for round_num in range(max_rounds):
            self.bulk_training(training_df=training_df, round_num=round_num, batch_size=batch_size,
                               priority_limiting=True, priority_setting=True)
            self.single_round_predict(round_num=round_num, batch_size=batch_size)

    def summary(self):
        """
        Summarize the current state of the run as a DataFrame.
        """
        if not self.summary:
            print('No logs present')
            return pd.DataFrame()

        summary = pd.DataFrame.from_dict(self.summary, orient='index')
        return summary

    def visualize(self, plot_type):
        """
        Visualize the prediction results of the run.
        """
        options = {
            'scatter':scatter
            }

        options[plot_type](self)
    
