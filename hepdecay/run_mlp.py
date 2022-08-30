from multiml import StoreGate, Saver
from multiml.agent import GridSearchAgent, RandomSearchAgent

from modules import MLPModel
from tasks import MyMLPTask
from metrics import graph_acc, edge_acc
from agent_metrics import MyAUCMetric
import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']

task_args = yml['task_args']
task_args['num_epochs'] = 5
task_args['batch_size'] = 2048
task_args['model'] = MLPModel
task_args['num_workers'] = 1
task_args['verbose'] = 0

task_args['metrics'] = ['loss', 'acc', 'lr']
task_args['output_var_names'] = 'pred_graphs'
task_args['true_var_names'] = 'label_graphs'
task_args['pred_var_names'] = None

agent_args = yml['agent_args']
agent_args['num_workers'] = 5
agent_args['num_trials'] = 1
agent_args['disable_tqdm'] = False
agent_args['metric'] = MyAUCMetric(var_names='pred_graphs label_graphs')

task_hps = dict(
    data_id = [
        'hepdecay_flat_high_ttbar_2hdm425-325_500000',
        'hepdecay_flat_low_ttbar_2hdm425-325_500000',
        'hepdecay_flat_high_ttbar_zprime1000_500000',
        'hepdecay_flat_low_ttbar_zprime1000_500000',
    ],
    model__layers = [3, 4, 5, 6], # [3, 4, 5, 6], 
    model__nodes = [32, 64, 128, 256], # [32, 64, 128, 256],
)

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args)
    sg.show_info()

    sv = Saver(save_dir='./saver.mlp', mode='zarr')

    task = MyMLPTask(**task_args)
    agent = GridSearchAgent(storegate=sg,
                            saver=sv,
                            task_scheduler=[[(task, task_hps)]],
                            **agent_args)
    agent.execute_finalize()
