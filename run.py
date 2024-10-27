from os import system
system('pip install mpl-toolkits seaborn matplotlib pandas scikit-learn')
codes = ['ModelFinal1_PC.py','ModelFinal1.py']

print('1)Per Capita\n2)Total\n')
i = int(input('Select the code you want to run: '))

system(f'python3 {codes[i-1]}')
