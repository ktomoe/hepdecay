from multiml import StoreGate, Saver
import yaml
import numpy as np

import xgboost as xgb

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']
#sg_args['data_id'] = 'hepdecay_flat_low_ttbar_2hdm425-325_500000'

configs = {
    'hepdecay_flat_low_ttbar_2hdm425-325_500000':  [(3, 5), (4, 5), (5, 5), (6, 5),],
    'hepdecay_flat_high_ttbar_2hdm425-325_500000': [(3, 5), (4, 5), (5, 5), (6, 5),],
    'hepdecay_flat_low_ttbar_zprime1000_500000':   [(3, 5), (4, 5), (5, 5), (6, 5),],
    'hepdecay_flat_high_ttbar_zprime1000_500000':  [(3, 5), (4, 5), (5, 5), (6, 5),],
}

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args)

    results = {}

    for data_id, max_depths in configs.items():
        sg.set_data_id(data_id)    
        sg.compile()
        sg.show_info()

        feature_train = sg.get_data('features', phase='train')
        label_train = sg.get_data('label_graphs', phase='train')
        feature_valid = sg.get_data('features', phase='valid')
        label_valid = sg.get_data('label_graphs', phase='valid')
        feature_test = sg.get_data('features', phase='test')
        label_test = sg.get_data('label_graphs', phase='test')

        dtrain = xgb.DMatrix(feature_train, label=label_train)
        dvalid = xgb.DMatrix(feature_valid, label=label_valid)
        dtest = xgb.DMatrix(feature_test, label=label_test)

        results[data_id] = []

        for (max_depth, num_iter) in max_depths:
            result_iter = []

            for ii in range(num_iter): 

                param = {'random_state': ii, 
                         'subsample': 0.8,
                         'max_depth': max_depth,
                         'eta': 1,
                         'eval_metric': 'auc',
                         'nthread':64}

                evallist = [(dvalid, 'eval'), (dtrain, 'train')]

                num_round = 100
                bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)

                ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(label_test, ypred)
                from sklearn.metrics import auc
                roc_auc = auc(fpr, tpr)

                result_iter.append(roc_auc)

            results[data_id].append(result_iter)

    for data_id, result in results.items():
        resutl = np.array(result)
        print ('-----')
        print (data_id)
        print (np.average(result, axis=1), np.std(result, axis=1))
