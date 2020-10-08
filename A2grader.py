run_my_solution = False
assignmentNumber = '2'

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A2mysolution import *
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

required_funcs = ['add_ones', 'make_weights', 'forward', 'backward', 'train_sgd', 'use', 'rmse',
                  'calc_standardize_parameters', 'standardize_X', 'standardize_T',
                  'unstandardize_T', 'forward_asig', 'backward_asig', 'train_sgd_asig', 'use_asig']



for func in required_funcs:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0


#### use ##################################################################

print('''
## Testing ####################################################################

    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}

    def print_layers(what, lst):
        print(f'{what}:')
        for (i, element) in enumerate(lst):
            print(f' Layer {i}:')
            print(f' {element}')

    print('X is')
    print(X)
    print_layers('Ws', Ws)
    print('stand_parms is')
    print(stand_parms)
    Ys = use(X, Ws, stand_parms)
''')

try:
    pts = 10
    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}

    def print_layers(what, lst):
        print(f'{what}:')
        for (i, element) in enumerate(lst):
            print(f' Layer {i}:')
            print(f' {element}')

    print('X is')
    print('', X)
    print_layers('Ws', Ws)
    print('stand_parms is')
    print('', stand_parms)
    Ys = use(X, Ws, stand_parms)
    # print_layers('Ys', Ys)
    
    correct_Ys = [np.array([[8.88178420e-16, 9.83674858e-01, 9.99864552e-01],
                            [1.97375320e-01, 9.95054754e-01, 9.99981668e-01],
                            [3.79948962e-01, 9.98507942e-01, 9.99997519e-01],
                            [5.37049567e-01, 9.99550366e-01, 9.99999664e-01]]),
                  np.array([[-3.22561051e-01,  6.32178901e-02,  4.30986476e-01,
                             6.95697105e-01],
                            [-4.01790666e-01, -1.27599476e-04,  4.01576644e-01,
                             6.91686573e-01],
                            [-4.70524536e-01, -6.02102999e-02,  3.71513128e-01,
                             6.86146821e-01],
                            [-5.25556964e-01, -1.11968562e-01,  3.44426847e-01,
                             6.80826586e-01]]),
                  np.array([[0.07247087],
                            [0.09337023],
                            [0.10716565],
                            [0.11581849]])]

    if all([np.allclose(cY, Y, rtol=0.01) for cY, Y in zip(correct_Ys, Ys)]):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values in Ys.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values in Ys.')
        print_layers('correct Ys values', correct_Ys)
        print_layers('Your Ys values are', Ys)
except Exception as ex:
    print(f'\n--- 0/{pts} points. use raised the exception\n')
    print(ex)



#### backward  ##################################################################
print('''
## Testing ####################################################################

    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}
    print('X is')
    print(X)
    print('T is')
    print(T)
    print_layers('Ws', Ws)
    print('stand_parms is')
    print(stand_parms)
    gradients = backward(X, T, Ws)
''')

try:
    pts = 10
    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}
    print('X is\n', X)
    print('T is\n', T)
    print_layers('Ws', Ws)
    print('stand_parms is\n', stand_parms)
    gradients = backward(X, T, Ws)

    correct_gradients = [np.array([[ 1.60447258e-02, -3.08375441e-03, -4.56974774e-05],
                                   [ 2.23767291e-01, -1.43644450e-02, -2.32945629e-04]]),
                         np.array([[-0.37104903,  0.        ,  0.49393342,  0.59482597],
                                   [-0.27543618,  0.        ,  0.33127065,  0.4035759 ],
                                   [-0.37311454,  0.        ,  0.49575972,  0.5971472 ],
                                   [-0.37106589,  0.        ,  0.49394889,  0.59484553]]),
                         np.array([[ 1.09720631],
                                   [-0.63996098],
                                   [-0.1765797 ],
                                   [ 0.35163518],
                                   [ 0.74251524]])]


    if all([np.allclose(cg, g, rtol=0.01) for cg, g in zip(correct_gradients, gradients)]):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values in gradients.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values in gradients.')
        print_layers('correct gradients values', correct_gradients)
        print_layers('Your gradients values are', gradients)
except Exception as ex:
    print(f'\n--- 0/{pts} points. backward raised the exception\n')
    print(ex)
    

#### train_sgd  ##################################################################
print('''
## Testing ####################################################################

    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}

    print('X is')
    print(X)
    print_layers('Ws', Ws)
    print('stand_parms is')
    print(stand_parms)
    Ys = use_asig(X, Ws, stand_parms)
''')

try:
    pts = 10
    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}

    print('X is\n', X)
    print_layers('Ws', Ws)
    print('stand_parms is\n', stand_parms)
    Ys = use_asig(X, Ws, stand_parms)
    # print_layers('Ys', Ys)
    
    correct_Ys = [np.array([[0.5       , 0.9168273 , 0.99183743],
                            [0.549834  , 0.95257413, 0.99698158],
                            [0.59868766, 0.97340301, 0.99888746],
                            [0.64565631, 0.98522597, 0.99959043]]),
                  np.array([[0.35959807, 0.46938325, 0.58221387, 0.68704854],
                            [0.35551243, 0.46796643, 0.58377151, 0.69101324],
                            [0.35088268, 0.46529809, 0.58348573, 0.6927973 ],
                            [0.34618066, 0.4621209 , 0.58230814, 0.69345436]]),
                  np.array([[-0.20164356],
                            [-0.19485722],
                            [-0.19090117],
                            [-0.1884819 ]])]

    if all([np.allclose(cY, Y, rtol=0.01) for cY, Y in zip(correct_Ys, Ys)]):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values in Ys.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values in Ys.')
        print_layers('correct Ys values', correct_Ys)
        print_layers('Your Ys values are', Ys)
except Exception as ex:
    print(f'\n--- 0/{pts} points. use_asig raised the exception\n')
    print(ex)



#### backward_asig  ##################################################################
print('''
## Testing ####################################################################

    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}
    print('X is')
    print(X)
    print('T is')
    print(T)
    print_layers('Ws', Ws)
    print('stand_parms is')
    print(stand_parms)

    gradients = backward_asig(X, T, Ws)
''')

try:
    pts = 10
    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)

    stand_parms = {'Xmeans': np.array([[0]]), 'Xstds': np.array([[1]]),
                   'Tmeans': np.array([[0]]), 'Tstds': np.array([[1]])}
    print('X is\n', X)
    print('T is\n', T)
    print_layers('Ws', Ws)
    print('stand_parms is\n', stand_parms)

    gradients = backward_asig(X, T, Ws)

    correct_gradients = [np.array([[ 0.00243401, -0.0020982 , -0.00073167],
                                   [ 0.0244516 , -0.00346653, -0.0033887 ]]),
                         np.array([[-0.09036161,  0.        ,  0.09801915,  0.17027297],
                                   [-0.06564802,  0.        ,  0.07098813,  0.12355229],
                                   [-0.09249953,  0.        ,  0.1002378 ,  0.17423368],
                                   [-0.09070056,  0.        ,  0.0983762 ,  0.17090459]]),
                         np.array([[0.80602904],
                                   [0.27318256],
                                   [0.36932491],
                                   [0.46940961],
                                   [0.56174678]])]

    if all([np.allclose(cg, g, rtol=0.01) for cg, g in zip(correct_gradients, gradients)]):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values in gradients.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values in gradients.')
        print_layers('correct gradients values', correct_gradients)
        print_layers('Your gradients values are', gradients)
except Exception as ex:
    print(f'\n--- 0/{pts} points. backward raised the exception\n')
    print(ex)
    

#### train_sgd_asig  ##################################################################
print('''
## Testing ####################################################################

    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)
    print('X is')
    print(X)
    print('T is')
    print(T)
    print_layers('Ws is', Ws)

    Ws, stand_parms, error_trace = train_sgd_asig(X, T, Ws, 0.1, 100)
''')

try:
    pts = 10
    X = np.arange(4).reshape(-1, 1) + 5
    T = np.array([1, 2, -3, -4]).reshape((-1, 1))
    Ws = make_weights(1, [3, 4], 1)
    for W in Ws:
        W[:] = np.linspace(-1, 1, W.size).reshape(W.shape)
    print('X is\n', X)
    print('T is\n', T)
    print_layers('Ws is', Ws)

    Ws, stand_parms, error_trace = train_sgd_asig(X, T, Ws, 0.5, 5000)

    correct_Ws = [np.array([[-0.37111225,  0.12240861, -1.44871785],
                            [ 1.07010709,  1.59683143,  5.63057907]]),
                  np.array([[-1.40681085, -0.72712667, -0.66044611, -0.25476951],
                            [-1.0844913 , -0.77391719, -0.22385811,  0.6335092 ],
                            [-0.26129973, -0.53239707,  0.20887749,  1.60869702],
                            [ 3.70190888,  1.90350124,  0.40761285, -0.71535807]]),
                  np.array([[ 0.76885524],
                            [-3.30633918],
                            [-1.7522539 ],
                            [ 0.02619815],
                            [ 2.12345964]])]
    correct_last_error =  5.075941137338459e-08

    if False:
        plt.figure(1)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(error_trace)
        plt.subplot(2, 1, 2)
        plt.plot(X, T, 'o')
        Ys = use_asig(X, Ws, stand_parms)
        plt.plot(X, Ys[-1])
    

    if all([np.allclose(cW, W, rtol=0.01) for cW, W in zip(correct_Ws, Ws)]):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values in Ws.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values in Ws.')
        print_layers('correct Ws values', correct_Ws)
        print_layers('Your Ws values are', Ws)

    pts = 10
    if abs(error_trace[-1] - correct_last_error) < 0.2:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct final error in error_trace.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values in error_trace.')
        print('correct final error_trace value is', correct_last_error)
        print_layers('Your final error_trace values is', error_trace[-1])
        
except Exception as ex:
    print(f'\n--- 0/{pts} points. train_sgd raised the exception\n')
    print(ex)
    










name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 60')
print('='*70)

print('''
__ / 10 points. Correct implementation of for loop to train neural nets with one, two, three
                and four hidden layers each with 4 hidden units. Train each for 10,000 epochs
                and a learning rate of 0.1.

__ / 10 points. Construction of pandas Dataframe to display results of above four loop.

__ / 10 points. Good discussion of the results from the four loop.  Use at least four sentences.

__ / 10 points. Good discussion of results you get with the above loop using the asymmetric sigmoid
                activation function.  Use at least six sentences.  In your discussion, compare differences
                and similarities between the results for tanh and asymmetric sigmoid.''')

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
