#!/usr/bin/env python3
# coding: utf-8

class FeaturesLauncher():

    def __init__(self,  db_path, meta_normalize=[True], meta_advance_features=[False], meta_advance_signals=[False], meta_history=[False], meta_layers=[[2048,1024,512,256,128,64,32]], convolutions=[0], activations=['relu'], meta_n=[126], meta_seeds=[0], epochss=[1], ss=[21], meta_n_past_features=[21], utilities=['f_drawdown']):
        print('Epochs ' + str(epochss))
        print('Seed ' + str(meta_seeds))
        print('Training period size ' + str(meta_n))
        print('Neural net layout ' + str(meta_layers))
        print('Advanced ' + str(meta_normalize))
        print('Advanced ' + str(meta_advance_features))
        print('Advanced ' + str(meta_advance_signals))
        print('History ' + str(meta_history))
        print('Normalized ' + str(meta_normalize))
        print('Rebalancing ' + str(ss))
        print('Activation ' + str(activations))
        print('Convolution' + str(convolutions))
        print('Utility' + str(utilities))
        print('n past features ' + str(meta_n_past_features))
        print('Rebalancing ' + str(ss))

        self.args = []
        self.counter = 1
        for utility in utilities:
            for n_past_features in meta_n_past_features:
                for convolution in convolutions:
                    for activation in activations:
                        for s in ss:
                            for epochs in epochss:
                                for normalize in meta_normalize:
                                    for whole_history in meta_history:
                                        for advance_feature in meta_advance_features:
                                            for advance_signal in meta_advance_signals:
                                                for seed in meta_seeds:
                                                    for layers in meta_layers:
                                                        for n in meta_n:
                                                            self.args.append((self,seed, utility, layers, epochs,
                                                                         n_past_features, n, s, whole_history,
                                                                         advance_feature, advance_signal, normalize,
                                                                         activation, convolution))
                                                            # self.args.append((seed, utility, param, layers, epochs,
                                                            #              n_past_features, n, s, whole_history,
                                                            #              advance_feature, advance_signal, normalize,
                                                            #              activation, convolution))
                                                            self.counter = self.counter + 1

        self.db_path = db_path
        self.runs = []
        self.totalRow = 0
        self.empty_runs_to_investigate = []



    def launchExplainer(self, toRun):
        if len(self.args)>1:
            raise Exception('Features explanation for one run only')
        featuresDf = toRun(*self.args[0])
        return featuresDf



