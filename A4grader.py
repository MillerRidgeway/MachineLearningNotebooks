run_my_solution = False
assignmentNumber = '4'

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A4mysolution import *
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

required_funcs = ['NeuralNetwork', 'NeuralNetworkClassifier', 'percent_correct', 'train_for_best_validation']

for func in required_funcs:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0


#### constructor ##################################################################

print('''
## Testing constructor ####################################################################

    # Linear network
    nnet = NeuralNetworkClassifier(2, [], 5)
    # Is isinstance(nnet, NeuralNetwork) True?
''')
      
try:
    pts = 5
    nnet = NeuralNetworkClassifier(2, [], 5)


    if isinstance(nnet, NeuralNetwork):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. NeuralNetworkClassifier is correctly of type NeuralNetwork')
    else:
        print(f'\n---  0/{pts} points. NeuralNetworkClassifier should be of type NeuralNetwork but it is not.')
        print(f'                  Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetworkClassifier raised the exception\n')
    print(ex)


print('''
## Testing constructor ####################################################################

    # Linear network
    nnet = NeuralNetworkClassifier(2, [], 5)
    W_shapes = [W.shape for W in nnet.Ws]
''')
      
try:
    pts = 5
    nnet = NeuralNetworkClassifier(2, [], 5)
    W_shapes = [W.shape for W in nnet.Ws]
    correct = [(3, 5)]

    if correct == W_shapes:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. W_shapes is correct value of {W_shapes}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values {W_shapes}.')
        print(f'                 Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetworkClassifier raised the exception\n')
    print(ex)
    

print('''
## Testing constructor ####################################################################

    G_shapes = [G.shape for G in nnet.Gs]
''')
      
try:
    pts = 5
    G_shapes = [G.shape for G in nnet.Gs]
    correct = [(3, 5)]

    if correct == G_shapes:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. G_shapes is correct value of {G_shapes}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values {G_shapes}.')
        print(f'                 Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Accessing nnet.Gs raised the exception\n')
    print(ex)



def print_layers(what, lst):
    print(f'{what}:')
    for (i, element) in enumerate(lst):
        print(f' Layer {i}:')
        print(f' {element}')



#### train  ##################################################################
print('''
## Testing train ####################################################################

    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)

    nnet = NeuralNetworkClassifier(2, [10, 5], 2)
    nnet.train(X, T, 1000, method='scg')

    Then check  nnet.get_error_trace()
''')

try:
    pts = 10
    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)

    nnet = NeuralNetworkClassifier(2, [10, 5], 2)
    nnet.train(X, T, 1000, method='scg')

    last_error = nnet.get_error_trace()[-1]
    correct = 0.9999999997132287

    if np.allclose(last_error, correct, rtol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct values in error_trace')
    else:
        print(f'\n---  0/{pts} points. Incorrect values in error_trace')
        print(f'                 Your value is {last_error[0]}, but it should be {correct[0]}')

except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.train or get_error_trace() raised the exception\n')
    print(ex)
    


#### train  ##################################################################
print('''
## Testing train ####################################################################

    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)

    nnet = NeuralNetworkClassifier(2, [10, 5], 2)
    nnet.train(X, T, 1000, method='scg')
    classes, prob = nnet.use(X)
''')

try:
    pts = 10
    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)

    nnet = NeuralNetworkClassifier(2, [10, 5], 2)
    nnet.train(X, T, 1000, method='scg')
    classes, prob = nnet.use(X)
    correct_classes = np.array([[0],
                                [1],
                                [1],
                                [0],
                                [1],
                                [0],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [0],
                                [0],
                                [0],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [1],
                                [0],
                                [1],
                                [0],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [0],
                                [0],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0],
                                [1],
                                [1],
                                [1],
                                [1]])
    correct_prob = np.array([[9.99999992e-01, 7.59567880e-09],
                             [6.09379897e-12, 1.00000000e+00],
                             [6.16799347e-12, 1.00000000e+00],
                             [1.00000000e+00, 1.84484302e-11],
                             [6.05233606e-12, 1.00000000e+00],
                             [1.00000000e+00, 1.83657451e-11],
                             [9.99999999e-01, 1.01771845e-09],
                             [6.16206667e-12, 1.00000000e+00],
                             [6.21859867e-12, 1.00000000e+00],
                             [6.15074193e-12, 1.00000000e+00],
                             [3.99126690e-09, 9.99999996e-01],
                             [6.13973057e-12, 1.00000000e+00],
                             [6.34006118e-12, 1.00000000e+00],
                             [6.69623630e-12, 1.00000000e+00],
                             [9.99999990e-01, 1.03292599e-08],
                             [2.37869969e-11, 1.00000000e+00],
                             [1.00000000e+00, 1.84001552e-11],
                             [6.05993969e-12, 1.00000000e+00],
                             [6.22902897e-12, 1.00000000e+00],
                             [6.19597131e-12, 1.00000000e+00],
                             [8.79563572e-12, 1.00000000e+00],
                             [1.00000000e+00, 1.83471767e-11],
                             [8.86341411e-12, 1.00000000e+00],
                             [6.19982345e-12, 1.00000000e+00],
                             [7.17023823e-12, 1.00000000e+00],
                             [6.08337885e-12, 1.00000000e+00],
                             [6.00976100e-12, 1.00000000e+00],
                             [6.17925409e-12, 1.00000000e+00],
                             [6.21834938e-12, 1.00000000e+00],
                             [6.73834222e-12, 1.00000000e+00],
                             [6.14666106e-12, 1.00000000e+00],
                             [4.59246732e-11, 1.00000000e+00],
                             [6.31250303e-12, 1.00000000e+00],
                             [1.00000000e+00, 2.10726806e-11],
                             [1.00000000e+00, 1.84254176e-11],
                             [9.99999995e-01, 5.03153122e-09],
                             [1.00000000e+00, 1.83400401e-11],
                             [6.03858734e-12, 1.00000000e+00],
                             [9.99999999e-01, 6.64406676e-10],
                             [6.27954235e-12, 1.00000000e+00],
                             [6.14730852e-12, 1.00000000e+00],
                             [6.36760007e-12, 1.00000000e+00],
                             [6.13209436e-12, 1.00000000e+00],
                             [6.07400369e-12, 1.00000000e+00],
                             [7.64388724e-12, 1.00000000e+00],
                             [1.00000000e+00, 3.14989200e-11],
                             [6.13168117e-12, 1.00000000e+00],
                             [6.22828809e-12, 1.00000000e+00],
                             [6.11464037e-12, 1.00000000e+00],
                             [6.22674041e-12, 1.00000000e+00],
                             [1.00000000e+00, 2.37389839e-11],
                             [6.18456985e-12, 1.00000000e+00],
                             [9.99999999e-01, 7.76463363e-10],
                             [6.49258323e-12, 1.00000000e+00],
                             [6.19079786e-12, 1.00000000e+00],
                             [6.16790631e-12, 1.00000000e+00],
                             [6.04745488e-12, 1.00000000e+00],
                             [6.06436604e-12, 1.00000000e+00],
                             [9.99999999e-01, 1.01121023e-09],
                             [6.45444772e-12, 1.00000000e+00],
                             [5.99896244e-12, 1.00000000e+00],
                             [6.22748566e-12, 1.00000000e+00],
                             [6.23598713e-12, 1.00000000e+00],
                             [6.00686383e-12, 1.00000000e+00],
                             [9.99999995e-01, 5.01621907e-09],
                             [6.19295344e-12, 1.00000000e+00],
                             [6.34158561e-12, 1.00000000e+00],
                             [9.99999999e-01, 1.34507129e-09],
                             [6.08930170e-12, 1.00000000e+00],
                             [1.00000000e+00, 9.14442270e-11],
                             [9.99999999e-01, 6.75326774e-10],
                             [6.17933303e-12, 1.00000000e+00],
                             [6.31919147e-12, 1.00000000e+00],
                             [6.10380087e-12, 1.00000000e+00],
                             [6.43049434e-12, 1.00000000e+00],
                             [9.99999999e-01, 7.39525262e-10],
                             [7.35737532e-12, 1.00000000e+00],
                             [9.99999999e-01, 6.52092127e-10],
                             [6.03407272e-12, 1.00000000e+00],
                             [1.53816705e-08, 9.99999985e-01],
                             [6.24589187e-12, 1.00000000e+00],
                             [6.09528221e-12, 1.00000000e+00],
                             [1.00000000e+00, 1.88113449e-11],
                             [6.16556430e-12, 1.00000000e+00],
                             [1.00000000e+00, 5.96743348e-11],
                             [9.99999999e-01, 6.95009925e-10],
                             [6.35281677e-12, 1.00000000e+00],
                             [9.22742377e-10, 9.99999999e-01],
                             [6.33723478e-12, 1.00000000e+00],
                             [9.99999999e-01, 6.31301186e-10],
                             [6.25323820e-12, 1.00000000e+00],
                             [6.01397830e-12, 1.00000000e+00],
                             [8.75021161e-12, 1.00000000e+00],
                             [6.18780945e-12, 1.00000000e+00],
                             [6.35301412e-12, 1.00000000e+00],
                             [1.00000000e+00, 1.85361500e-11],
                             [6.17229233e-12, 1.00000000e+00],
                             [6.12588196e-12, 1.00000000e+00],
                             [6.01477036e-12, 1.00000000e+00],
                             [6.04895529e-12, 1.00000000e+00]])


    if np.allclose(classes, correct_classes, rtol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct values in classes')
    else:
        print(f'\n---  0/{pts} points. Incorrect values in classes')
        print(f'                 Your value is\n {classes}.')
        print(f'                 Correct value is\n {correct_classes}')

    if np.allclose(prob, correct_prob, rtol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct values in prob')
    else:
        print(f'\n---  0/{pts} points. Incorrect values in prob')
        print(f'                 Your value is\n {prob}.')
        print(f'                 Correct value is\n {correct_prob}')


except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.train or use raised the exception\n')
    print(ex)
    


######################################################################
print('''
## Testing percent_correct ####################################################################

    Y = np.array([1, 2, 3, 1, 2, 3]).reshape(-1, 1)
    T = np.array([1, 2, 3, 3, 2, 1]).reshape(-1, 1)
    pc = percent_correct(Y, T)
''')

try:
    pts = 5

    Y = np.array([1, 2, 3, 1, 2, 3]).reshape(-1, 1)
    T = np.array([1, 2, 3, 3, 2, 1]).reshape(-1, 1)
    pc = percent_correct(Y, T)
    correct = 100 * 2/3

    if np.allclose(pc, correct, rtol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct percent_correct.')
    else:
        print(f'\n---  0/{pts} points. Incorrect percent_correct')
        print(f'                  Correct value is {correct}')
        print(f'                  Your value is {pc}')
        
except Exception as ex:
    print(f'\n--- 0/{pts} points. percent_correct raised the exception\n')
    print(ex)
    


    
######################################################################
print('''
## Testing train_for_best_validation ####################################################################

    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    np.random.shuffle(X)
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)

    Xtrain = X[:50, :]
    Ttrain = T[:50, :]
    Xval = X[50:75, :]
    Ttest = T[50:75, :]
    Xtest = X[75:, :]
    Ttest = T[75:, :]

    nnet, epoch, train_accuracy, val_accuracy = train_for_best_validation(Xtrain, Ttrain,  Xval, Tval, 
                                                                          400, 10, [10, 10], method='scg') 


''')

try:
    pts = 5

    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    np.random.shuffle(X)
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)

    Xtrain = X[:50, :]
    Ttrain = T[:50, :]
    Xval = X[50:75, :]
    Tval = T[50:75, :]
    Xtest = X[75:, :]
    Ttest = T[75:, :]

    np.random.seed(42)
    nnet, epoch, train_accuracy, val_accuracy = train_for_best_validation(Xtrain, Ttrain,  Xval, Tval, 
                                                                          400, 5, [10, 10], method='scg') 
    if epoch == 15:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct best epoch.')
    else:
        print(f'\n---  0/{pts} points. Incorrect best epoch')
        print('                  Correct value is 15')
        print('                  Your value is {epoch}')
                
    if train_accuracy == 98.0:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct train_accuracy.')
    else:
        print(f'\n---  0/{pts} points. Incorrect train_accuracy')
        print('                  Correct value is 98.0')
        print('                  Your value is {train_accuracy}')
                
    if val_accuracy == 100.0:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct val_accuracy.')
    else:
        print(f'\n---  0/{pts} points. Incorrect val_accuracy')
        print('                  Correct value is 100.0')
        print('                  Your value is {val_accuracy}')
                
        
except Exception as ex:
    print(f'\n--- 0/{pts} points. train_for_best_validation raised the exception\n')
    print(ex)
    


name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 65')
print('='*70)

print('''
__ / 5 points. Correctly downloaded and read the MNIST data.

__ / 10 points. Correctly applied train_for_best_validation function
                to MNIST data.

__ / 5 points. Experimented with different values of parameters as 
               arguments to train_for_best_validation.

__ / 5 points. Show confusion matrix for best neural network.

__ / 10 points. Described results with at least 10 sentences.''')

print()
print('='*70)
print(f'{name} Results and Discussion Grade is ___ / 35')
print('='*70)


print()
print('='*70)
print(f'{name} FINAL GRADE is  _  / 100')
print('='*70)


print('''
Extra Credit: Earn one point of extra credit for repeating the above
experiments for another classification data set.''')

print(f'\n{name} EXTRA CREDIT is 0 / 1')


if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

    
