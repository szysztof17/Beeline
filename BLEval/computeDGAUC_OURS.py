import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"lines.linewidth": 2}, palette = "deep", style = "ticks")
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from itertools import product, permutations, combinations, combinations_with_replacement
from tqdm import tqdm
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, r
import time

def PRROC(dataDict, inputSettings, directed=True, selfEdges=False, plotFlag=True):
    '''
    Computes areas under the precision-recall and ROC curves
    for a given dataset for each algorithm.

    :param directed:     A flag to indicate whether to treat predictions as directed edges (directed = True) or undirected edges (directed = False).
    :type directed: bool
    :param selfEdges:     A flag to indicate whether to include self-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool
    :param plotFlag:     A flag to indicate whether or not to save PR and ROC plots.
    :type plotFlag: bool

    :returns:
              - AUPRC: A dictionary containing AUPRC values for each algorithm
              - AUROC: A dictionary containing AUROC values for each algorithm
    '''
    trueEdgesDF = pd.read_csv(str(inputSettings.datadir)+'/'+ dataDict['name'] +
                                '/' +dataDict['trueEdges'],
                                sep = ',',
                                header = 0, index_col = None)
    outDir = Path("outputs") / inputSettings.datadir.relative_to("inputs") / dataDict['name']


    # Initialize data dictionaries
    precisionDict = {}
    recallDict = {}
    FPRDict = {}
    TPRDict = {}
    AUPRC = {}
    AUPRC_sklearn = {}

    AUROC = {}
    AUPRC_RATIO = {}
    '''
    AUPRC_prroc_r_dict = {}
    AUPRC_r_classic_dict = {}
    '''

    if directed:
        for algo in tqdm(inputSettings.algorithms,
                                    total=len(inputSettings.algorithms), unit=" Algorithms"):

            # check if the output rankedEdges file exists
            ranked_edges_path = outDir / algo[0] / 'rankedEdges.csv'
            if ranked_edges_path.exists():
                # Initialize Precision
                predDF = pd.read_csv(ranked_edges_path, sep='\t', header=0, index_col=None)
                print(f'Working on {algo[0]}, dataset: {str(ranked_edges_path).split("/")[-3]}')
                #precisionDict[algo[0]], recallDict[algo[0]], FPRDict[algo[0]], TPRDict[algo[0]], AUPRC[algo[0]], AUROC[algo[0]], AUPRC_prroc_r_dict[algo[0]], AUPRC_r_classic_dict[algo[0]] = computeScores(trueEdgesDF, predDF, directed=True, selfEdges=selfEdges, algo_name = algo[0])
                precisionDict[algo[0]], recallDict[algo[0]], FPRDict[algo[0]], TPRDict[algo[0]], AUPRC[algo[0]], AUROC[algo[0]], AUPRC_RATIO[algo[0]], AUPRC_sklearn[algo[0]] = computeScores(trueEdgesDF, predDF, directed=True, selfEdges=selfEdges, algo_name = algo[0])
                print(f'AUPRC_RATIO[algo[0]]: {AUPRC_RATIO[algo[0]]}')

            else:
                print(f'{ranked_edges_path} does not exist. Skipping...')
            PRName = 'PRplot'
            ROCName = 'ROCplot'
    else:
        for algo in tqdm(inputSettings.algorithms,
                                    total=len(inputSettings.algorithms), unit=" Algorithms"):

            # check if the output rankedEdges file exists
            ranked_edges_path = outDir / algo[0] / 'rankedEdges.csv'
            if ranked_edges_path.exists():
                # Initialize Precision
                predDF = pd.read_csv(ranked_edges_path, sep='\t', header=0, index_col=None)

                precisionDict[algo[0]], recallDict[algo[0]], FPRDict[algo[0]], TPRDict[algo[0]], AUPRC[algo[0]], AUROC[algo[0]] = computeScores(trueEdgesDF, predDF, directed=False, selfEdges=selfEdges)

            else:
                print(f'{ranked_edges_path} does not exist. Skipping...')

            PRName = 'uPRplot'
            ROCName = 'uROCplot'
    print(f'calculated scores with computeScores from BLEval_mine')

    if plotFlag:
        # Make PR curves
        legendList = []
        for key in recallDict.keys():
            sns.lineplot(recallDict[key], precisionDict[key], ci=None)
            legendList.append(f'{key} (AUPRC = {AUPRC[key]:.2f})')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(legendList)
        plt.savefig(outDir / f'{PRName}.pdf')
        plt.savefig(outDir / f'{PRName}.png')
        plt.clf()

        # Make ROC curves
        legendList = []
        for key in recallDict.keys():
            sns.lineplot(FPRDict[key], TPRDict[key], ci=None)
            legendList.append(f'{key} (AUROC = {AUROC[key]:.2f})')

        plt.plot([0, 1], [0, 1], linewidth=1.5, color='k', linestyle='--')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(legendList)
        plt.savefig(outDir / f'{ROCName}.pdf')
        plt.savefig(outDir / f'{ROCName}.png')
        plt.clf()

    return AUPRC, AUROC, AUPRC_RATIO


def sort_Gene1_Gene2(df):
    df['minGene'] = df[['Gene1', 'Gene2']].min(axis=1)
    df['maxGene'] = df[['Gene1', 'Gene2']].max(axis=1)
    df['Gene1'] = df['minGene']
    df['Gene2'] = df['maxGene']
    df = df.drop(columns=['minGene', 'maxGene'])
    return df

def computeScores(trueEdgesDF, predEdgeDF,
                            directed = True, selfEdges = True, verbose_time = False, algo_name = 'a'):
    start_time = time.time()  # Track the start time

    start_step = time.time()
    #print(f'trueEdgesDF.columns: {trueEdgesDF.columns}')
    unique_genes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    #pred_unique = np.unique(predEdgeDF.loc[:, ['Gene1', 'Gene2']])
    #all_genes = np.union1d(unique_genes, pred_unique)

    if verbose_time:
        print(f"Time for extracting unique genes: {time.time() - start_step:.4f} seconds")
    start_step = time.time()

    # construct a generator of all possible graph edges
    if directed:
        edge_generator = product(unique_genes, repeat=2) if selfEdges else permutations(unique_genes, r=2)
    else:
        edge_generator = combinations_with_replacement(unique_genes, r=2) if selfEdges else combinations(unique_genes, r=2)
    if verbose_time:
        print(f"Time for constructing edge generator: {time.time() - start_step:.4f} seconds")

    # master DataFrame of all possible edges
    start_step = time.time()

    outDF = pd.DataFrame.from_records(edge_generator, columns = ['Gene1', 'Gene2'])
    if verbose_time:
        print(f"Time for creating all egdes DataFrame: {time.time() - start_step:.4f} seconds")

    del edge_generator
    start_step = time.time()
    if not directed:
        outDF = sort_Gene1_Gene2(outDF)
        if verbose_time:
            print(f"Time for sorting master DataFrame: {time.time() - start_step:.4f} seconds")
    #outDF.set_index(['Gene1', 'Gene2'], inplace=True)
    #outDF['Gene1'] = outDF['Gene1'].astype(gene_dtype)
    #outDF['Gene2'] = outDF['Gene2'].astype(gene_dtype)


    # DataFrame of true edges
    start_step = time.time()

    teDF = trueEdgesDF[['Gene1', 'Gene2']]
    if not directed:
        teDF = sort_Gene1_Gene2(teDF)
    # these true edges should have no duplicates if directed==True, but let's be on the safe side here
    teDF = teDF.drop_duplicates()
    #teDF.set_index(['Gene1', 'Gene2'], inplace=True)

    #teDF['Gene1'] = teDF['Gene1'].astype(gene_dtype)
    #teDF['Gene2'] = teDF['Gene2'].astype(gene_dtype)

    teDF['TrueEdges'] = 1
    #teDF.set_index(['Gene1', 'Gene2'], inplace=True)

    if verbose_time:
        print(f"Time for creating true edges DataFrame: {time.time() - start_step:.4f} seconds")

    #print(teDF.head())
    #duplicates = teDF[teDF.duplicated(subset=['Gene1', 'Gene2'], keep=False)]
    #print(f'found {len(duplicates)} duplicates')
    #print(duplicates)
    # merge with the master DataFrame
    outDF = pd.merge(outDF, teDF, on = ['Gene1', 'Gene2'], how='left', validate='one_to_one')
    #outDF = pd.merge(outDF, teDF, left_index = True, right_index = True, how='left', validate='one_to_one')

    outDF['TrueEdges'].fillna(0, inplace = True)

    # DataFrame of predicted edges
    start_step = time.time()

    peDF = predEdgeDF[['Gene1', 'Gene2', 'EdgeWeight']]

    # use the absolute value of the predicted edge weight
    peDF['PredEdges'] = abs(peDF['EdgeWeight'])
    if not directed:
        peDF = sort_Gene1_Gene2(peDF)
    #peDF.set_index(['Gene1', 'Gene2'], inplace=True)

    #peDF['Gene1'] = peDF['Gene1'].astype(gene_dtype)
    #peDF['Gene2'] = peDF['Gene2'].astype(gene_dtype)
    # these predicted edges should have no duplicates if directed==True, but let's be on the safe side here
    #peDF = peDF.groupby(level=0)['PredEdges'].max().to_frame()
    peDF = peDF.groupby(['Gene1', 'Gene2']).agg({'PredEdges': 'max'})
    #print(peDF.head())
    if verbose_time:
        print(f"Time for creating predicted edges DataFrame: {time.time() - start_step:.4f} seconds")

    # merge with the master DataFrame
    start_step = time.time()

    outDF = pd.merge(outDF, peDF, on = ['Gene1', 'Gene2'], how='left', validate='one_to_one')
    #outDF = pd.merge(outDF, peDF, left_index = True, right_index = True, how='left', validate='one_to_one')

    outDF['PredEdges'].fillna(0, inplace = True)
    #print(outDF.head())
    if verbose_time:
        print(f"Time for merging predicted edges: {time.time() - start_step:.4f} seconds")
    outDF['Name'] = outDF['Gene1'] + '|' + outDF['Gene2']
    #outDF[['Name', 'PredEdges', 'TrueEdges']].to_csv(f'{algo_name}_pr_label_data.csv', index=True)

    prroc = importr('PRROC')

    prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(outDF['PredEdges'].values)), 
              weights_class0 = FloatVector(list(outDF['TrueEdges'].values)))#, curve = True)
    random_prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(outDF['PredEdges'].values)), 
              weights_class0 = FloatVector(list(outDF['TrueEdges'].values)), rand_compute = True)

    # Calculate PR and ROC curves using scikit-learn
    start_step = time.time()
    prec, recall, _ = precision_recall_curve(y_true=outDF['TrueEdges'], probas_pred=outDF['PredEdges'], pos_label=1)
    avg_prec = average_precision_score(outDF['TrueEdges'], outDF['PredEdges'])
    fpr, tpr, _ = roc_curve(y_true=outDF['TrueEdges'], y_score=outDF['PredEdges'], pos_label=1)
    auprc_score = auc(recall, prec) # AUPRC using trapezoidal rule in sklearn
    auroc_score = auc(fpr, tpr) 

    auprc_score_prroc_r = prCurve[2][0] # AUPRC from PRROC package in R
    auprc_score_r_classic = prCurve[1][0] # AUPRC from PRROC package in R, but classic integration

    random_prCurve_score = random_prCurve[3][1][0] # AUPRC from PRROC package in R
    print(f'AUPRC RATIO: {auprc_score_prroc_r / random_prCurve_score}')
    print(f'AUPRC: {auprc_score_prroc_r}, random_prCurve_score: {random_prCurve_score}')
    #print(f'auprc_score: {auprc_score}, average_precision_score: {avg_prec}, auprc_score_prroc_r: {auprc_score_prroc_r}, auprc_score_r_classic: {auprc_score_r_classic} ')

    if verbose_time:
        print(f"Time for calculating PR and ROC curves: {time.time() - start_step:.4f} seconds")
    '''
    # Plot and save PR curves
    plt.figure(figsize=(8, 6))
    plt.plot(recall, prec, label='Scikit-learn PR')
    plt.plot(r_recall, r_precision, label='R PRROC PR', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.savefig(f'pr_curve_comparison_{algo_name}.png')  # Save the plot
    plt.close() # close the figure so that next figure is not plotted on top of it.

    import pickle
    pickle_file = f'object_{algo_name}.pkl'
    with open(pickle_file, 'wb') as file:
        pickle.dump(prCurve, file)
    print(f'LETs see hat this is: prcurve saved to {pickle_file}')
    '''


    #print(f'Calculated AUPRC:\nwith PRROC R package: {prCurve[2][0]}\nwith scikit-learn: {auprc_score} \n and with R, but classic integration {prCurve[1][0]}')

    print('Used the BLeval_mine')
    return prec, recall, fpr, tpr, auprc_score_prroc_r, auroc_score, auprc_score_prroc_r / random_prCurve_score, auprc_score      ###, auprc_score_prroc_r, auprc_score_r_classic


