from multiml.storegate import StoreGate
import yaml
import numpy as np

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

data_id = 'hepdecay'
# data_id, processes, target_events, num_nodes

outputs = [
    ['graph',     ['ttbar', '2hdm425-325'], [500000], 6],
    ['random',    ['ttbar', '2hdm425-325'], [500000], 6],
    ['flat_low',  ['ttbar', '2hdm425-325'], [500000], 6],
    ['flat_high', ['ttbar', '2hdm425-325'], [500000], 6],
    ['graph',     ['ttbar', 'zprime1000'],  [500000], 6],
    ['random',    ['ttbar', 'zprime1000'],  [500000], 6],
    ['flat_low',  ['ttbar', 'zprime1000'],  [500000], 6],
    ['flat_high', ['ttbar', 'zprime1000'],  [500000], 6],
]

##############################################################################

def reshape_data(data, highlevel, num_nodes, task):
    num_batch = data.shape[0]
    data = data.reshape(num_batch, 6, 5)

    data[:, :, 0] = np.ma.log10(data[:, :, 0]).filled(0.) # pt
    data[:, :, 3] = np.ma.log10(data[:, :, 3]).filled(0.) # mass


    if task == 'flat_low':
        data = data.reshape(num_batch, 30)

    elif task == 'flat_high':
        data = data.reshape(num_batch, 30)
        highlevel = np.ma.log10(highlevel).filled(0.)
     
        data = np.concatenate([data, highlevel], axis=1)

    return data

def fill(sg, data_id_org, task, processes, max_event, num_nodes):
    data_id = f'{data_id_org}_{task}_{processes[0]}_{processes[1]}_{max_event}'

    for phase in ('train', 'test', 'valid'):
        if phase != 'train':
            max_event = 50000

        sg.set_data_id(data_id_org)
        features_low0 = sg.get_data(processes[0]+'_features_low', phase)[:max_event]
        features_low1 = sg.get_data(processes[1]+'_features_low', phase)[:max_event]
        features_high0 = sg.get_data(processes[0]+'_features_high', phase)[:max_event]
        features_high1 = sg.get_data(processes[1]+'_features_high', phase)[:max_event]

        data0 = reshape_data(features_low0, features_high0, num_nodes, task)
        data1 = reshape_data(features_low1, features_high1, num_nodes, task)

        graph_label0 = np.zeros(len(data0), dtype='i8')
        graph_label1 = np.ones(len(data1), dtype='i8')

        edge_label0 = sg.get_data(processes[0]+'_edges', phase)[:max_event]
        edge_label1 = sg.get_data(processes[1]+'_edges', phase)[:max_event]

        if 'random' in task:
            edge_label0 = np.random.randint(0, 6, (len(data0), 36))
            edge_label1 = np.random.randint(0, 6, (len(data1), 36))

        edge_label0 = edge_label0.astype('i8')
        edge_label1 = edge_label1.astype('i8')

        edge_label0 = np.identity(6)[edge_label0]
        edge_label1 = np.identity(6)[edge_label1]

        sg.set_data_id(data_id)
        sg.delete_data('features', phase)
        sg.delete_data('label_graphs', phase)
        sg.delete_data('label_edges', phase)

        sg.add_data('features', data0, phase)
        sg.add_data('features', data1, phase)
        sg.add_data('label_graphs', graph_label0, phase)
        sg.add_data('label_graphs', graph_label1, phase)
        sg.add_data('label_edges', edge_label0, phase)
        sg.add_data('label_edges', edge_label1, phase)

    sg.compile()
    sg.shuffle()

    sg.show_info()

if __name__ == "__main__":
    sg = StoreGate(**yml['sg_args_a'])

    for task, output, max_events, num_nodes in outputs:
        sg.set_data_id(data_id)
        sg.compile()

        for max_event in max_events:
            fill(sg, data_id, task, output, max_event, num_nodes)
