import logging
from datetime import datetime


def SetLogging(logname: str, append: bool = False):
	mode = 'a' if append else 'w'
	logging.basicConfig(format='%(asctime)s.%(msecs)03d;%(levelname)s;{%(module)s};[%(funcName)s];%(thread)d-%(process)d;%(message)s',
						datefmt='%d/%m/%Y %I:%M:%S',
						handlers=[logging.StreamHandler(), logging.FileHandler(logname, mode=mode)],
						level=logging.INFO)

if __name__ == '__main__':
	now_str = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
	SetLogging(f'wqu_capstone_{now_str}.log', False)
