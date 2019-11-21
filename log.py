from datetime import datetime
import os

class Log:
    def __init__(self, exp_name):
        self.file = os.path.join(exp_name, 'log_loss.txt')
        with open(self.file, 'w') as f:
            f.write('==== %s ====' % datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    def info(self, message):
        print(message)
        with open(self.file, 'a') as f:
            f.write(message + '\n')
