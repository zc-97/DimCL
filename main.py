from SELFRec import SELFRec
from util.conf import ModelConf
from util.svd import get_svd
import os
if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['LightGCN','DirectAU','MF']
    ssl_graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL','MixGCF','LightGCL','DimCL','SimGRACE','DimXsimCL','FinalSimDimCL']
    sequential_baselines= ['SASRec']
    ssl_sequential_models = ['CL4SRec','DuoRec','BERT4Rec']

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)
    # model = input('Please enter the model you want to run:')
    device = 'gpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    # model = 'MF'
    # model = 'LightGCN'
    # model = 'LightGCL'
    # model = 'NCL'
    # model = 'SGL'
    # model = 'SimGRACE'
    # model = 'SimGCL'
    # model = 'XSimGCL'
    # model = 'DimCL'  # LightGCN + DimCL
    # model = 'FinalSimDimCL'  # SimGCL + DimCL
    model = 'DimXsimCL'  # XSimGCL + DimCL


    import time

    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
        conf.config['device'] = device
    else:
        print('Wrong model name!')
        exit(-1)
    if model =='LightGCL':
        dataset = conf.config['training.set'].strip().split('/')[2]
        conf.config['svd'] = get_svd(dataset,device)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
