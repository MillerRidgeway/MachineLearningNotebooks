run_my_solution = False

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A1mysolution import *
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')

    assignmentNumber = '1'

    import subprocess, glob, pathlib
    nb_name = '*-A{}.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         nb_name.format(assignmentNumber), '--stdout'], stdout=outputFile)
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


    
for func in ['polynomial_model', 'polynomial_gradient', 'gradient_descent_adam', 'rmse']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0



######################################################################
print('''\nTesting
    X = np.array([1, -2, 3, -4, 5, -8, 9, -10]).reshape((-1, 1))
    W = np.ones((4, 1))
    Y = polynomial_model(X, W)
''')

try:
    pts = 20
    X = np.array([1, -2, 3, -4, 5, -8, 9, -10]).reshape((-1, 1))
    W = np.ones((4, 1))
    Y = polynomial_model(X, W)
    correct_Y = np.array([[   4.],
                          [  -5.],
                          [  40.],
                          [ -51.],
                          [ 156.],
                          [-455.],
                          [ 820.],
                          [-909.]])
    if np.allclose(Y, correct_Y, 1e-4):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Y should be')
        print(correct_Y)
        print(f'                       Your values are')
        print(Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. polynomial_model raised the exception\n')
    print(ex)
        

######################################################################
print('''\nTesting
    X = np.array([1, -2, 3, -4, 5, -8, 9, -10]).reshape((-1, 1))
    W = np.ones((4, 1))
    Y = polynomial_model(X, W)
    T = np.array([[   4.2],
                  [  -4.8],
                  [  40.2],
                  [ -50.8],
                  [ 156.2],
                  [-454.8],
                  [ 820.2],
                  [-908.8]])
    gradient = polynomial_gradient(X, T, W)
''')

try:
    pts = 20
    X = np.array([1, -2, 3, -4, 5, -8, 9, -10]).reshape((-1, 1))
    W = np.ones((4, 1))
    Y = polynomial_model(X, W)
    T = np.array([[   4.2],
                  [  -4.8],
                  [  40.2],
                  [ -50.8],
                  [ 156.2],
                  [-454.8],
                  [ 820.2],
                  [-908.8]])
    gradient = polynomial_gradient(X, T, W)
    correct_gradient = np.array([[ -0.4],
                                 [  0.3],
                                 [-15. ],
                                 [ 35.1]])
    if np.allclose(gradient, correct_gradient, 1e-4):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. gradient should be')
        print(correct_gradient)
        print(f'                       Your values are')
        print(gradient)
except Exception as ex:
    print(f'\n--- 0/{pts} points. polynomial_gradient raised the exception\n')
    print(ex)
        
######################################################################
print('''\nTesting
    X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
    T = (X - 5) * 0.05 + 0.002 * (X - 8)**2
    W = np.zeros((5, 1))
    rho = 0.01
    n_steps = 100
    W, _, _ = gradient_descent_adam(polynomial_model, polynomial_gradient, rmse,
                                    X, T, W, rho, n_steps)
''')

try:
    pts = 20
    X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
    T = (X - 5) * 0.05 + 0.002 * (X - 8)**2
    W = np.zeros((5, 1))
    rho = 0.01
    n_steps = 100
    W, _, _ = gradient_descent_adam(polynomial_model, polynomial_gradient, rmse,
                                    X, T, W, rho, n_steps)
    correct_W = np.array([[-2.28854790e-03],
                          [-1.08662984e-04],
                          [ 1.93905941e-04],
                          [ 1.61351414e-04],
                          [ 6.66501091e-05]])
    if np.allclose(W, correct_W, 1e-4):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. W should be')
        print(correct_W)
        print(f'                       Your values are')
        print(W)
except Exception as ex:
    print(f'\n--- 0/{pts} points. train raised the exception\n')
    print(ex)




name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 60')
print('='*70)

print('''
__ / 10 Reading in air quality data and plotting it correctly.

__ / 10 points. Applying the Adam optimizer to the air quality data using your polynomial model and gradient correctly.

__ / 10 points.  Plotting the resulting error curve in one graph and plotting the model predictions on top of data
        correctly.  Also describe what you observe in these two graphs with at least five total sentences.

__ / 10 points. Show and describe results for three different values of n_powers and also for three different values of n_steps,
                Describe what you see with at least eight sentences.''')

print()
print('='*70)
print(f'{name} Results and Discussion Grade is ___ / 40')
print('='*70)


print()
print('='*70)
print(f'{name} FINAL GRADE is  _  / 100')
print('='*70)


if run_my_solution:
    from A1mysolution import *
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')
