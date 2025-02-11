import os

def func(input):
    output = input.split()[-1]
    return output

if __name__ == '__main__':
    with open('input.txt', 'w') as f:
        f.write(prompt)
    subprocess.call(['python', 'temp.py'])
