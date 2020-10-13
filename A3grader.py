run_my_solution = False
assignmentNumber = '3'

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A3mysolution import *
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')


    import subprocess, glob, pathlib
    n = assignmentNumber
    nb_names = glob.glob(f'*-A{n}-[0-9].ipynb') + glob.glob(f'*-A{n}.ipynb')
    nb_names = np.unique(nb_names)
    nb_names = sorted(nb_names, key=os.path.getmtime)
    if len(nb_names) > 1:
        print(f'More than one ipynb file found: {nb_names}. Using first one.')
    filename = nb_names[0]
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         filename, '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ImportFrom) and
            not isinstance(node, ast.ClassDef)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *

required_funcs = ['NeuralNetworks']



for func in required_funcs:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0


#### constructor ##################################################################

print('''
## Testing constructor ####################################################################

    nnet = NeuralNetwork(2, [5, 4], 3)
    W_shapes = [W.shape for W in nnet.Ws]
''')
      
try:
    pts = 5
    nnet = NeuralNetwork(2, [5, 4], 3)
    W_shapes = [W.shape for W in nnet.Ws]
    correct = [(3, 5), (6, 4), (5, 3)]

    if correct == W_shapes:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. W_shapes is correct value of {W_shapes}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values {W_shapes}.')
        print(f'                   Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork raised the exception\n')
    print(ex)


print('''
## Testing constructor ####################################################################

    G_shapes = [G.shape for G in nnet.Gs]
''')
      
try:
    pts = 5
    G_shapes = [G.shape for G in nnet.Gs]
    correct = [(3, 5), (6, 4), (5, 3)]

    if correct == G_shapes:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. G_shapes is correct value of {G_shapes}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values {G_shapes}.')
        print(f'                   Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Accessing nnet.Gs raised the exception\n')
    print(ex)


print('''
## Testing constructor ####################################################################

    nnet.Ws[0][2, 0] = 100.0
    # Does nnet.Ws[0][2, 0] == nnet.all_weights[10]
''')
      
try:
    pts = 10
    nnet.Ws[0][2, 0] = 100.0

    if nnet.Ws[0][2, 0] == nnet.all_weights[10]:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. nnet.Ws[0][2, 0] equals nnet.all_weights[10]')
    else:
        print(f'\n---  0/{pts} points. nnet.Ws[0][2, 0] does not equal nnet.all_weights[10]')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Raises exception\n')
    print(ex)









print('''
## Testing constructor ####################################################################

    G_shapes = [G.shape for G in nnet.Gs]
''')
      
try:
    pts = 10
    G_shapes = [G.shape for G in nnet.Gs]
    correct = [(3, 5), (6, 4), (5, 3)]

    if correct == G_shapes:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. G_shapes is correct value of {G_shapes}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values {G_shapes}.')
        print(f'                   Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetworks raised the exception\n')
    print(ex)





def print_layers(what, lst):
    print(f'{what}:')
    for (i, element) in enumerate(lst):
        print(f' Layer {i}:')
        print(f' {element}')



#### train  ##################################################################
print('''
## Testing train ####################################################################

    Now with two outputs (columns in T)

    X = np.arange(20).reshape(-1, 2) + 5
    T = np.hstack((X[:, 0:1] * 0.4, (X[:, 1:2] / 10) ** 3))

    np.random.seed(123)
    nnet = NeuralNetwork(2, [5, 4, 3], 2)
    nnet.train(X, T, 1, 0.01, method='sgd')

    Check nnet.Gs

    Then check  nnet.get_error_trace()

''')

try:
    pts = 10
    X = np.arange(20).reshape(-1, 2) + 5
    T = np.hstack((X[:, 0:1] * 0.4, (X[:, 1:2] / 10) ** 3))

    np.random.seed(123)
    nnet = NeuralNetwork(2, [5, 4, 3], 2)
    nnet.train(X, T, 1, 0.01, method='sgd')

    correct_Gs =  [np.array([[-0.00414994, -0.00388862,  0.01489535,  0.00228742, -0.00178074],
                             [-0.0097602 , -0.00575799,  0.0398138 ,  0.00607998, -0.00425533],
                             [-0.0097602 , -0.00575799,  0.0398138 ,  0.00607998, -0.00425533]]),
                   np.array([[-0.00863046, -0.02109961,  0.00922772, -0.01999294],
                             [ 0.00538085,  0.01282735, -0.00635168,  0.01214879],
                             [-0.01592876, -0.03819697,  0.01868124, -0.03657976],
                             [-0.00112788, -0.00258563,  0.00150607, -0.00242617],
                             [ 0.01286295,  0.03081336, -0.01479321,  0.02895545],
                             [ 0.00438864,  0.01043688, -0.00522767,  0.00988769]]),
                   np.array([[-0.0233086 , -0.07065335, -0.05894822],
                             [-0.01184439, -0.04260322, -0.02930195],
                             [-0.0037676 , -0.01921643, -0.00829576],
                             [ 0.02195175,  0.08546413,  0.05410017],
                             [ 0.01271116,  0.0545411 ,  0.02959401]]),
                   np.array([[-0.17551841, -0.20508025],
                             [ 0.07486432,  0.07249534],
                             [ 0.0251894 ,  0.02175168],
                             [-0.05886119, -0.04970528]])]

    if all([np.allclose(cG, G, rtol=0.01) for cG, G in zip(correct_Gs, nnet.Gs)]):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct values in nnet.Gs')
    else:
        print(f'\n---  0/{pts} points. Incorrect values in nnet.Gs')
        print_layers('correct Gs values', correct_Gs)
        print_layers('Your nnet.Gs values are', nnet.Gs)

    pts = 10
    correct_error_trace = [1.033635920599668]
    if np.allclose(correct_error_trace, nnet.get_error_trace(), rtol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct error_trace {nnet.get_error_trace()}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect value.')
        print('Correct error_trace is', correct_error_trace)
        print('Your returned error_trace is', nnet.get_error_trace())
        
except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.get_error_trace() raised the exception\n')
    print(ex)
    


#### train all methods  ##################################################################
print('''
## Testing train all methods ####################################################################

    Now with two outputs (columns in T)

    X = np.arange(20).reshape(-1, 2) + 5
    T = np.hstack((X[:, 0:1] * 0.4, (X[:, 1:2] / 10) ** 3))

    np.random.seed(123)
    nnet_sgd = NeuralNetwork(2, [5, 4, 3], 2)
    nnet_sgd.train(X, T, 1000, 0.01, method='sgd')

    np.random.seed(123)
    nnet_adam = NeuralNetwork(2, [5, 4, 3], 2)
    nnet_adam.train(X, T, 1000, 0.01, method='adam')

    np.random.seed(123)
    nnet_scg = NeuralNetwork(2, [5, 4, 3], 2)
    nnet_scg.train(X, T, 1000, None, method='scg')  # learning_rate is None, not used by scg

    def rmse(Y, T):
        return np.sqrt(np.mean((T - Y)**2))

    rmse_sgd = rmse(nnet_sgd.use(X), T)
    rmse_adam = rmse(nnet_adam.use(X), T)
    rmse_scg = rmse(nnet_scg.use(X), T)

    Check [rmse_sgd, rmse_adam, rmse_scg]
''')

try:
    pts = 10

    X = np.arange(20).reshape(-1, 2) + 5
    T = np.hstack((X[:, 0:1] * 0.4, (X[:, 1:2] / 10) ** 3))

    np.random.seed(123)
    nnet_sgd = NeuralNetwork(2, [5, 4, 3], 2)
    nnet_sgd.train(X, T, 1000, 0.01, method='sgd')

    np.random.seed(123)
    nnet_adam = NeuralNetwork(2, [5, 4, 3], 2)
    nnet_adam.train(X, T, 1000, 0.01, method='adam')

    np.random.seed(123)
    nnet_scg = NeuralNetwork(2, [5, 4, 3], 2)
    nnet_scg.train(X, T, 1000, None, method='scg')  # learning_rate is None, not used by scg

    def rmse(Y, T):
        return np.sqrt(np.mean((T - Y)**2))

    rmse_sgd = rmse(nnet_sgd.use(X), T)
    rmse_adam = rmse(nnet_adam.use(X), T)
    rmse_scg = rmse(nnet_scg.use(X), T)

    correct_rmse = (0.7901712599763487, 0.01880579461876818, 0.0030167886304268508)
    
    if np.allclose(correct_rmse, [rmse_sgd, rmse_adam, rmse_scg], rtol = 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct rmse values.')
    else:
        print(f'\n---  0/{pts} points. Incorrect rmse values')
        print('                        correct rmse values are', correct_rmse)
        print('                         Your rmse values are', [rmse_sgd, rmse_adam, rmse_scg])

        
except Exception as ex:
    print(f'\n--- 0/{pts} points. train or use raised the exception\n')
    print(ex)
    


    

name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 60')
print('='*70)

print('''
__ / 10 points. Reading data and defining Xtrain, Ttrain, Xtest, Ttest.

__ / 10 points. Experiments with variety of values for n_hiddens_list, n_epochs, and learning_rate
                for the three optimization methods.

__ / 10 points. Plots of your best results each of the three methods.

__ / 10 points. Good discussion of results you get, using at least 10 sentences.''')

print()
print('='*70)
print(f'{name} Results and Discussion Grade is ___ / 40')
print('='*70)


print()
print('='*70)
print(f'{name} FINAL GRADE is  _  / 100')
print('='*70)


if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')
