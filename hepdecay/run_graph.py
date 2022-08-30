from multiml import StoreGate, Saver
from multiml.agent import GridSearchAgent

from modules import GraphModel
from tasks import MyGraphTask
from callbacks import get_dgl
from metrics import graph_acc, edge_acc
from agent_metrics import MyAUCMetric
import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']

task_args = yml['task_args']
task_args['num_epochs'] = 100
task_args['batch_size'] = 2048
task_args['model'] = GraphModel
task_args['num_workers'] = 2
task_args['verbose'] = 0

task_args['dataset_args'] = dict(callbacks=[get_dgl])
task_args['metrics'] += [edge_acc, graph_acc]

agent_args = yml['agent_args']
agent_args['num_workers'] = [0, 1, 2, 3, 4]
agent_args['num_trials'] = 5
agent_args['disable_tqdm'] = False
agent_args['metric'] = MyAUCMetric(var_names='pred_graphs label_graphs')

task_hps = dict(
    data_id = [
        'hepdecay_graph_ttbar_zprime1000_500000',
        'hepdecay_graph_ttbar_2hdm425-325_500000',
        'hepdecay_random_ttbar_zprime1000_500000',
        'hepdecay_random_ttbar_2hdm425-325_500000',
    ],
    loss_weights = [
        [0., 1.],  [1., 1.],  [2., 1.],  [3., 1.],
        [4., 1.],  [5., 1.],  [6., 1.],  [7., 1.],  
        [8., 1.],  [9., 1.],  [10., 1.], [11., 1.],
        [12., 1.], [13., 1.], [14., 1.], [15., 1.],
        [16., 1.], [17., 1.], [18., 1.], [19., 1.],
        [20., 1.], [21., 1.], [22., 1.], [23., 1.],
        [24., 1.], [25., 1.], [26., 1.], [27., 1.],
        [28., 1.], [29., 1.], [30., 1.], [31., 1.],
        [32., 1.], [33., 1.], [34., 1.], [35., 1.],
        [36., 1.], [37., 1.], [38., 1.], [39., 1.],
        [40., 1.], [41., 1.], [42., 1.], [43., 1.],
        [44., 1.], [45., 1.], [46., 1.], [47., 1.],
        [48., 1.], [49., 1.], [50., 1.], [51., 1.],
        [52., 1.], [53., 1.], [54., 1.], [55., 1.],
        [56., 1.], [57., 1.], [58., 1.], [59., 1.],
        [60., 1.], [61., 1.], [62., 1.], [63., 1.],
    ],
    model__features = ['gat'],
    model__nodes = [256], # [128, 256, 512],
    model__edge_layers = [4],  # [3, 4, 5],
    model__graph_layers = [2], # [2, 3, 4],
    model__num_heads = [2], # [2, 4],
)

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args) 
    sg.show_info()

    sv = Saver(save_dir='./saver.gat', mode='zarr')

    task = MyGraphTask(**task_args)
    agent = GridSearchAgent(storegate=sg,
                            saver=sv,
                            task_scheduler=[[(task, task_hps)]],
                            **agent_args)
    agent.execute_finalize()
